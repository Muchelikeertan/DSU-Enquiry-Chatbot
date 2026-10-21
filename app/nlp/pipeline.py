"""NLP Pipeline Orchestrator for DSU College Enquiry Chatbot.

Coordinates:
1. Text Preprocessing
2. Longest-Match Entity Extraction
3. Multi-Turn Context & Slot Tracking (6-Turn TTL)
4. Intent Classification (TF-IDF + Logistic Regression)
5. Local FAQ Retrieval & Form Policy Enforcement
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.nlp.classifier import IntentClassifier, IntentPrediction
from app.nlp.context_tracker import ContextTracker, ConversationSlotContext
from app.nlp.entity_extractor import EntityExtractor, ExtractedEntities
from app.nlp.faq_retriever import FAQMatchResult, FAQRetriever
from app.nlp.preprocessor import TextPreprocessor
from app.nlp.query_analyzer import QueryAnalysis, QueryAnalyzer
from app.retrieval.dsu_retriever import DSURetriever, RetrievedEvidence

logger = logging.getLogger(__name__)


@dataclass
class ChatbotNLPResponse:
    """Structured response from the NLP Pipeline."""
    response_text: str
    intent: str
    confidence: float
    handler: str
    entities: ExtractedEntities
    context_slots: Dict[str, Any]
    show_enquiry_form: bool = False
    form_policy: str = "never"
    form_prefill: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None
    status: str = "source_checked"
    is_ambiguous: bool = False
    needs_fallback: bool = False
    fallback_handler: Optional[str] = None
    quick_replies: List[str] = field(default_factory=list)
    used_llm: bool = False


class NLPPipeline:
    """End-to-end NLP processing pipeline for DSU chatbot queries."""

    def __init__(
        self,
        intents_data: Dict[str, Any],
        college_data: Dict[str, Any],
        faq_data: Dict[str, Any],
        intent_confidence_threshold: float = 0.60,
        faq_confidence_threshold: float = 0.65,
        context_ttl_turns: int = 6,
        llm_handler: Optional[Any] = None,
        retriever: Optional[Any] = None,
    ):
        self.intents_data = intents_data
        self.college_data = college_data
        self.faq_data = faq_data

        # Initialize NLP Components
        self.preprocessor = TextPreprocessor()
        self.entity_extractor = EntityExtractor(
            college_data=college_data,
            preprocessor=self.preprocessor,
        )
        self.context_tracker = ContextTracker(ttl_turns=context_ttl_turns)
        self.classifier = IntentClassifier(
            confidence_threshold=intent_confidence_threshold,
            preprocessor=self.preprocessor,
        )
        self.classifier.fit(intents_data)

        self.faq_retriever = FAQRetriever(
            confidence_threshold=faq_confidence_threshold,
            preprocessor=self.preprocessor,
            faq_data=faq_data,
        )

        # Cache metadata
        self.intent_map = {i["tag"]: i for i in intents_data.get("intents", []) if "tag" in i}
        self.enquiry_cfg = college_data.get("enquiry", {})
        self.fee_fallback_msg = self.enquiry_cfg.get(
            "fee_fallback_message",
            "Sorry, I couldn't find verified fee details for this course for the 2026-27 academic year. Please fill out the enquiry form below."
        )

        # Initialize Dynamic Retriever (official DSU sources)
        self.retriever = retriever if retriever is not None else DSURetriever()

        # Initialize LLM Handler (modular fallback integration)
        if llm_handler is not None:
            self.llm_handler = llm_handler
        else:
            try:
                from app.config import settings
                from app.llm.handler import GroqLLMHandler
                self.llm_handler = GroqLLMHandler(
                    api_key=settings.groq_api_key,
                    model=settings.groq_model,
                    timeout=settings.groq_timeout_seconds,
                    max_retries=settings.groq_max_retries,
                )
            except Exception:
                self.llm_handler = None

    @classmethod
    def from_files(
        cls,
        intents_path: Path,
        college_info_path: Path,
        faq_info_path: Path,
        **kwargs,
    ) -> "NLPPipeline":
        """Loads canonical JSON files from disk and initializes the NLP pipeline."""
        with open(intents_path, "r", encoding="utf-8") as f:
            intents_data = json.load(f)
        with open(college_info_path, "r", encoding="utf-8") as f:
            college_data = json.load(f)
        with open(faq_info_path, "r", encoding="utf-8") as f:
            faq_data = json.load(f)

        return cls(intents_data, college_data, faq_data, **kwargs)

    def _get_sca_course_quick_replies(self) -> List[str]:
        """Returns quick reply suggestions for SCA courses."""
        return [c.get("name", c.get("id")) for c in self.college_data.get("sca", {}).get("courses", [])]

    def process_query(self, message: str, session_id: str = "default_session") -> ChatbotNLPResponse:
        """Processes a single user message through the complete NLP pipeline."""
        # 1. Step turn counter in context tracker
        self.context_tracker.step_turn(session_id)

        # 2. Extract academic entities
        entities = self.entity_extractor.extract(message)

        # 3. Analyze query structure, specific fee categories, academic year, and multi-part requirements
        norm_message = self.preprocessor.normalize(message)
        query_analysis = QueryAnalyzer.analyze(norm_message, entities=entities)
        query_analysis.raw_query = message

        # 4. Update session slots if entities are present (current message overrides stored context)
        if entities.course_id:
            self.context_tracker.update_slots(
                session_id=session_id,
                course_id=entities.course_id,
                school_id=entities.school_id,
                specialisation_text=entities.specialisation_text,
            )
        elif entities.school_id:
            self.context_tracker.update_slots(
                session_id=session_id,
                school_id=entities.school_id,
            )

        active_ctx = self.context_tracker.get_or_create_session(session_id)
        active_ctx.add_message("user", message)

        # 5. Classify intent
        prediction: IntentPrediction = self.classifier.predict(message)
        active_intent = prediction.tag
        trigger_form = False

        # Intent keyword/rule overrides for queries where TF-IDF classifier has low confidence or misclassifies
        q_lower = message.lower()
        if "scholarship" in q_lower or "concession" in q_lower:
            active_intent = "entrance_exams_scholarships"
        elif "dean" in q_lower and any(w in q_lower for w in ["sca", "computer", "application", "department", "school"]):
            active_intent = "sca_dean"
        elif QueryAnalyzer.is_fee_types_comparison_query(message) or query_analysis.is_comparison:
            active_intent = "sca_fees"
        elif any(k in q_lower for k in ["hostel", "residential room", "room charges", "mess charges", "meals for", "meals"]):
            active_intent = "hostel_facilities"

        # 6. Handle Context Follow-Ups (follow_up_*)
        if prediction.requires_context or active_intent.startswith("follow_up_"):
            resolved_intent, bundle, trigger_form = self.context_tracker.resolve_follow_up(
                session_id=session_id,
                follow_up_tag=active_intent,
                new_course_id=entities.course_id,
            )
            active_intent = resolved_intent

        # Check course-specific intents without context or entity:
        # According to intents.json context_rules.no_context_clarification,
        # asking a course question without naming a course routes to ask_which_course.
        course_specific_intents = {
            "sca_fees", "other_department_fees", "sca_course_duration",
            "sca_eligibility", "sca_location"
        }
        if active_intent in course_specific_intents and not active_ctx.course_id and not entities.course_id:
            if QueryAnalyzer.is_fee_types_comparison_query(message) or query_analysis.is_comparison:
                pass
            else:
                active_intent = "ask_which_course"

        intent_meta = self.intent_map.get(active_intent, {})
        handler = intent_meta.get("handler", prediction.handler)
        fallback_handler = intent_meta.get("fallback_handler", prediction.fallback_handler)

        # 6. Prefill payload for Enquiry Form
        form_prefill = {
            "course_name": active_ctx.course_id or (entities.course_id if entities else None),
            "school_or_department": active_ctx.school_id or (entities.school_id if entities else None),
            "question": message,
        }

        # Case A: If context tracker triggered form due to after_failed_lookup
        if trigger_form:
            resp_msg = (
                "This information is currently unverified or requires administrative confirmation. "
                "Please fill out the enquiry form below, and the university admissions team will assist you."
            )
            active_ctx.add_message("assistant", resp_msg)
            return ChatbotNLPResponse(
                response_text=resp_msg,
                intent=active_intent,
                confidence=prediction.confidence,
                handler="enquiry_form",
                entities=entities,
                context_slots={
                    "course_id": active_ctx.course_id,
                    "school_id": active_ctx.school_id,
                    "specialisation_text": active_ctx.specialisation_text,
                },
                show_enquiry_form=True,
                form_policy="always",
                form_prefill=form_prefill,
                status="dynamic_lookup_required",
            )

        # Case B: Context clarification requested (ask_which_course or ask_what_to_know)
        if active_intent == "ask_which_course":
            responses = intent_meta.get("responses", ["Which course are you interested in?"])
            resp_msg = responses[0] if responses else "Which course are you interested in?"
            active_ctx.add_message("assistant", resp_msg)
            return ChatbotNLPResponse(
                response_text=resp_msg,
                intent="ask_which_course",
                confidence=prediction.confidence,
                handler="canned_response",
                entities=entities,
                context_slots={
                    "course_id": active_ctx.course_id,
                    "school_id": active_ctx.school_id,
                },
                quick_replies=self._get_sca_course_quick_replies(),
            )

        if active_intent == "ask_what_to_know":
            q_clean = message.lower()
            if query_analysis.is_comparison or len(entities.course_ids) >= 2 or any(k in q_clean for k in ["difference", "compare", "curriculum", "career", "choose", "versus", " vs "]):
                active_intent = "sca_courses"
                handler = "llm"
                fallback_handler = "llm"
            else:
                responses = intent_meta.get("responses", ["What would you like to know about this course?"])
                resp_msg = responses[0] if responses else "What would you like to know about this course?"
                active_ctx.add_message("assistant", resp_msg)
                return ChatbotNLPResponse(
                    response_text=resp_msg,
                    intent="ask_what_to_know",
                    confidence=prediction.confidence,
                    handler="canned_response",
                    entities=entities,
                    context_slots={
                        "course_id": active_ctx.course_id,
                        "school_id": active_ctx.school_id,
                    },
                    quick_replies=["Fees", "Eligibility", "Duration", "Location"],
                )

        # Case C: Canned Responses (greeting, thanks, goodbye, capabilities)
        if handler == "canned_response":
            responses = intent_meta.get("responses", [])
            reply = responses[0] if responses else "Hello! How can I assist you with DSU admissions today?"
            active_ctx.add_message("assistant", reply)
            return ChatbotNLPResponse(
                response_text=reply,
                intent=active_intent,
                confidence=prediction.confidence,
                handler=handler,
                entities=entities,
                context_slots={
                    "course_id": active_ctx.course_id,
                    "school_id": active_ctx.school_id,
                },
                quick_replies=["SCA Courses", "Admissions 2026", "Campus Details", "Enquiry Form"],
            )

        # Case D: Enquiry intent
        if handler == "enquiry_form" or active_intent == "enquiry":
            resp_msg = "Please fill out the enquiry form so your question can be directed to the appropriate university team."
            active_ctx.add_message("assistant", resp_msg)
            return ChatbotNLPResponse(
                response_text=resp_msg,
                intent="enquiry",
                confidence=prediction.confidence,
                handler="enquiry_form",
                entities=entities,
                context_slots={
                    "course_id": active_ctx.course_id,
                    "school_id": active_ctx.school_id,
                },
                show_enquiry_form=True,
                form_policy="always",
                form_prefill=form_prefill,
                status="source_checked",
            )

        # Case E: Local retrieval (local or local_then_llm)
        retrieval_query = message
        if active_ctx.course_id and not entities.course_id:
            retrieval_query = f"{message} {active_ctx.course_id}"

        faq_match: Optional[FAQMatchResult] = self.faq_retriever.retrieve(
            query=retrieval_query,
            predicted_intent=active_intent,
            course_id=active_ctx.course_id or entities.course_id,
            school_id=active_ctx.school_id or entities.school_id,
        )

        is_sufficient = QueryAnalyzer.is_local_faq_sufficient(
            faq_match=faq_match,
            query=message,
            analysis=query_analysis,
        )

        if faq_match and faq_match.is_confident and faq_match.answer and is_sufficient:
            # Update context with verified answer details
            self.context_tracker.update_slots(
                session_id=session_id,
                topic_intent=active_intent,
                last_faq_id=faq_match.faq_id,
                last_answer_status=faq_match.status,
            )
            active_ctx.add_message("assistant", faq_match.answer)

            return ChatbotNLPResponse(
                response_text=faq_match.answer,
                intent=active_intent,
                confidence=prediction.confidence,
                handler=handler,
                entities=entities,
                context_slots={
                    "course_id": active_ctx.course_id,
                    "school_id": active_ctx.school_id,
                    "topic_intent": active_intent,
                },
                show_enquiry_form=faq_match.show_enquiry_form,
                form_policy=faq_match.form_policy,
                form_prefill=form_prefill if faq_match.show_enquiry_form else {},
                source=faq_match.source,
                status=faq_match.status,
                used_llm=False,
            )

        # Case F: Grounded LLM Fallback with Dynamic Official DSU Retrieval
        # Triggers whenever local records do not answer or provide incomplete details
        retrieved_evidence = []
        if self.retriever is not None:
            try:
                # Convert ConversationSlotContext to plain dict (retriever expects Dict[str, Any])
                retriever_context = {
                    "course_id": active_ctx.course_id,
                    "school_id": active_ctx.school_id,
                    "topic_intent": active_ctx.topic_intent,
                }
                retrieved_evidence = self.retriever.search(message, context=retriever_context)
                if len(entities.course_ids) >= 2 or query_analysis.is_comparison:
                    for cid in entities.course_ids:
                        extra_ev = self.retriever.search(f"{cid} eligibility duration fee", context=retriever_context)
                        for ev in extra_ev:
                            if not any(e.url == ev.url for e in retrieved_evidence):
                                retrieved_evidence.append(ev)
            except Exception as e:
                logger.warning("Dynamic retrieval error: %s", type(e).__name__)
                retrieved_evidence = []

        if self.llm_handler is not None:
            configured_policy = None
            if query_analysis.specific_fee_category in ("management_quota", "nri_quota", "cet_quota", "comedk_quota"):
                configured_policy = "always"
            elif active_intent == "other_department_fees":
                configured_policy = "always"

            llm_resp = self.llm_handler.generate_grounded_response(
                query=message,
                intent=active_intent,
                course_id=active_ctx.course_id or entities.course_id,
                school_id=active_ctx.school_id or entities.school_id,
                college_data=self.college_data,
                form_policy=configured_policy,
                retrieved_evidence=retrieved_evidence,
                conversation_history=active_ctx.history,
                query_analysis=query_analysis,
            )
            # Update context
            self.context_tracker.update_slots(
                session_id=session_id,
                topic_intent=active_intent,
                last_answer_status=llm_resp.status,
            )
            active_ctx.add_message("assistant", llm_resp.text)

            primary_src = llm_resp.primary_source
            if not primary_src and llm_resp.source_urls:
                primary_src = llm_resp.source_urls[0]
            if not primary_src:
                primary_src = f"Groq LLM ({llm_resp.model})" if not llm_resp.is_fallback else "Local Fallback"

            return ChatbotNLPResponse(
                response_text=llm_resp.text,
                intent=active_intent,
                confidence=prediction.confidence,
                handler=handler,
                fallback_handler=fallback_handler,
                entities=entities,
                context_slots={
                    "course_id": active_ctx.course_id,
                    "school_id": active_ctx.school_id,
                    "topic_intent": active_intent,
                },
                show_enquiry_form=llm_resp.show_enquiry_form,
                form_policy=llm_resp.form_policy,
                form_prefill=form_prefill if llm_resp.show_enquiry_form else {},
                source=primary_src,
                status=llm_resp.status,
                used_llm=not llm_resp.is_fallback,
                needs_fallback=llm_resp.is_fallback,
                is_ambiguous=prediction.is_ambiguous,
            )

        # Fallback when LLM handler is unavailable: synthesize from retrieved evidence if present
        if retrieved_evidence:
            # Critical fee-category protection: do not substitute general fees for specific quota/fee queries
            if query_analysis.specific_fee_category:
                cat_lower = query_analysis.specific_fee_category.replace("_", " ")
                evidence_text = " ".join([getattr(e, "content", "") for e in retrieved_evidence]).lower()
                evidence_has_cat = any(term in evidence_text for term in cat_lower.split())
                if not evidence_has_cat:
                    prog_name = entities.course_name or (active_ctx.course_id.upper() if active_ctx.course_id else "the programme")
                    resp_text = (
                        f"For the confirmed {cat_lower} fee for {prog_name}, please contact the DSU Admissions Cell. "
                        "You can submit an enquiry below to obtain the applicable amount."
                    )
                    active_ctx.add_message("assistant", resp_text)
                    return ChatbotNLPResponse(
                        response_text=resp_text,
                        intent=active_intent,
                        confidence=prediction.confidence,
                        handler=handler,
                        fallback_handler=fallback_handler,
                        entities=entities,
                        context_slots={
                            "course_id": active_ctx.course_id,
                            "school_id": active_ctx.school_id,
                            "topic_intent": active_intent,
                        },
                        show_enquiry_form=True,
                        form_policy="always",
                        form_prefill=form_prefill,
                        source=retrieved_evidence[0].url if retrieved_evidence else None,
                        status="source_checked_partial",
                        used_llm=False,
                        needs_fallback=True,
                    )

            top = retrieved_evidence[0]
            title = getattr(top, "title", "DSU Official Records")
            url = getattr(top, "url", "https://www.dsu.edu.in")
            content = getattr(top, "content", "").strip()
            lines = [l.strip() for l in content.split("\n") if l.strip()]
            lead = lines[0] if lines else f"Official DSU records confirm the following for {message}:"
            bullets = [l.lstrip("-*• ") for l in lines[1:5] if len(l.strip()) > 5]
            parts = [f"**Answer:** {lead}"]
            if bullets:
                parts.append("**Key details:**\n" + "\n".join(f"- {b}" for b in bullets))
            parts.append(f"**Official sources:**\n- [{title}]({url})")
            resp_text = "\n\n".join(parts)
            active_ctx.add_message("assistant", resp_text)
            return ChatbotNLPResponse(
                response_text=resp_text,
                intent=active_intent,
                confidence=prediction.confidence,
                handler=handler,
                fallback_handler=fallback_handler,
                entities=entities,
                context_slots={
                    "course_id": active_ctx.course_id,
                    "school_id": active_ctx.school_id,
                    "topic_intent": active_intent,
                },
                show_enquiry_form=False,
                form_policy="on_failure",
                source=url,
                status="source_checked",
                used_llm=False,
                needs_fallback=False,
            )

        # Case G: Fallback when neither LLM nor retrieved evidence is available
        if query_analysis.specific_fee_category:
            cat_name = query_analysis.specific_fee_category.replace("_", " ")
            prog_name = entities.course_name or (active_ctx.course_id.upper() if active_ctx.course_id else "this programme")
            fallback_text = (
                f"For the confirmed {cat_name} fee for {prog_name}, please contact the DSU Admissions Cell. "
                "You can submit an enquiry below to obtain the applicable amount."
            )
        elif active_intent in ("sca_fees", "other_department_fees"):
            fallback_text = self.fee_fallback_msg
        else:
            fallback_text = (
                "I could not verify that detail directly from the official records. "
                "Please use the enquiry form below or contact the university admissions office directly."
            )
        active_ctx.add_message("assistant", fallback_text)
        return ChatbotNLPResponse(
            response_text=fallback_text,
            intent=active_intent,
            confidence=prediction.confidence,
            handler=handler,
            fallback_handler=fallback_handler,
            entities=entities,
            context_slots={
                "course_id": active_ctx.course_id,
                "school_id": active_ctx.school_id,
            },
            show_enquiry_form=True,
            form_policy="always",
            form_prefill=form_prefill,
            status="dynamic_lookup_required",
            needs_fallback=True,
            is_ambiguous=prediction.is_ambiguous,
        )
