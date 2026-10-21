"""Conversation Context and Slot Tracking for DSU College Enquiry Chatbot.

Implements multi-turn conversation state tracking across a 6-turn TTL.
Maintains slots: course_id, specialisation_text, school_id, topic_intent,
last_faq_id, and last_answer_status.
Resolves pronoun cues ('it', 'that course') and maps follow_up_* intents
to target intents according to context_rules in intents.json.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class ConversationSlotContext:
    """Active slot values for a user session."""
    session_id: str
    course_id: Optional[str] = None
    specialisation_text: Optional[str] = None
    school_id: Optional[str] = None
    topic_intent: Optional[str] = None
    last_faq_id: Optional[str] = None
    last_answer_status: Optional[str] = None
    turn_count: int = 0
    inactive_turns: int = 0
    max_ttl_turns: int = 6
    history: List[Dict[str, str]] = field(default_factory=list)

    def is_expired(self) -> bool:
        """Returns True if context has exceeded the TTL turn limit."""
        return self.inactive_turns >= self.max_ttl_turns

    def add_message(self, role: str, content: str) -> None:
        """Records a message to the conversation history, retaining recent turns."""
        if content:
            self.history.append({"role": role, "content": content})
            if len(self.history) > 12:
                self.history = self.history[-12:]

    def clear_slots(self) -> None:
        """Clears all contextual slots upon TTL expiration."""
        self.course_id = None
        self.specialisation_text = None
        self.school_id = None
        self.topic_intent = None
        self.last_faq_id = None
        self.last_answer_status = None
        self.inactive_turns = 0


class ContextTracker:
    """Manages multi-turn conversation sessions and resolves follow-up queries."""

    NON_TOPIC_INTENTS: Set[str] = {
        "greeting",
        "goodbye",
        "thanks",
        "chatbot_capabilities",
        "enquiry",
        "fallback",
        "ask_which_course",
        "ask_what_to_know",
    }

    CUE_WORDS: List[str] = [
        "it", "its", "that", "this", "that course", "this course",
        "the same", "there", "that one", "this one"
    ]

    def __init__(self, ttl_turns: int = 6):
        self.ttl_turns = ttl_turns
        self.sessions: Dict[str, ConversationSlotContext] = {}

    def get_or_create_session(self, session_id: str) -> ConversationSlotContext:
        """Retrieves active session or initializes a fresh one."""
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationSlotContext(
                session_id=session_id,
                max_ttl_turns=self.ttl_turns,
            )
        ctx = self.sessions[session_id]
        if ctx.is_expired():
            ctx.clear_slots()
        return ctx

    def step_turn(self, session_id: str) -> ConversationSlotContext:
        """Increments turn counters and checks TTL expiration."""
        ctx = self.get_or_create_session(session_id)
        ctx.turn_count += 1
        ctx.inactive_turns += 1
        if ctx.is_expired():
            ctx.clear_slots()
        return ctx

    def update_slots(
        self,
        session_id: str,
        course_id: Optional[str] = None,
        school_id: Optional[str] = None,
        specialisation_text: Optional[str] = None,
        topic_intent: Optional[str] = None,
        last_faq_id: Optional[str] = None,
        last_answer_status: Optional[str] = None,
    ) -> ConversationSlotContext:
        """Updates slot values according to the update rules in intents.json."""
        ctx = self.get_or_create_session(session_id)

        # 1. Course update rule:
        # If the message names a course, set course_id and school_id; clear specialisation unless given.
        if course_id:
            ctx.course_id = course_id
            ctx.inactive_turns = 0
            if school_id:
                ctx.school_id = school_id
            elif course_id.startswith("sca_"):
                ctx.school_id = "computer_applications"

            if specialisation_text:
                ctx.specialisation_text = specialisation_text
            else:
                ctx.specialisation_text = None

        # 2. School-only update rule:
        # If message names only a school, set school_id and clear course_id
        elif school_id and not course_id:
            ctx.school_id = school_id
            ctx.course_id = None
            ctx.specialisation_text = None
            ctx.inactive_turns = 0

        # 3. Topic intent rule:
        # After answering a non-follow-up intent, set topic_intent.
        if topic_intent:
            if topic_intent not in self.NON_TOPIC_INTENTS and not topic_intent.startswith("follow_up_"):
                ctx.topic_intent = topic_intent
                ctx.inactive_turns = 0

        # 4. Status and FAQ rule:
        if last_faq_id:
            ctx.last_faq_id = last_faq_id
        if last_answer_status:
            ctx.last_answer_status = last_answer_status

        return ctx

    def resolve_follow_up(
        self,
        session_id: str,
        follow_up_tag: str,
        new_course_id: Optional[str] = None,
    ) -> Tuple[str, Optional[List[str]], bool]:
        """Resolves follow_up_* intents to concrete target intents based on context.

        Returns:
            Tuple of:
            - resolved_target_intent: str
            - bundle_intents: Optional[List[str]] (if follow_up_more_details returns a bundle)
            - trigger_enquiry_form: bool (True if after_failed_lookup policy triggers)
        """
        ctx = self.get_or_create_session(session_id)

        # Check 'after_failed_lookup' rule:
        # If last_answer_status is not_verified and student asks for more, show enquiry form.
        if ctx.last_answer_status in ("not_verified", "dynamic_lookup_required") and follow_up_tag == "follow_up_more_details":
            return "enquiry", None, True

        # Determine active course scope
        active_cid = new_course_id or ctx.course_id
        is_sca = bool(active_cid and active_cid.startswith("sca_"))
        is_other = bool(active_cid and not active_cid.startswith("sca_"))
        has_school_only = bool(ctx.school_id and not active_cid)

        # 1. follow_up_fee
        if follow_up_tag == "follow_up_fee":
            if is_sca:
                return "sca_fees", None, False
            elif is_other:
                return "other_department_fees", None, False
            else:
                return "ask_which_course", None, False

        # 2. follow_up_eligibility
        elif follow_up_tag == "follow_up_eligibility":
            if is_sca:
                return "sca_eligibility", None, False
            elif is_other:
                return "admissions_eligibility", None, False
            else:
                return "ask_which_course", None, False

        # 3. follow_up_duration
        elif follow_up_tag == "follow_up_duration":
            if is_sca:
                return "sca_course_duration", None, False
            elif is_other or has_school_only:
                return "other_department_courses", None, False
            else:
                return "ask_which_course", None, False

        # 4. follow_up_location
        elif follow_up_tag == "follow_up_location":
            if is_sca:
                return "sca_location", None, False
            elif is_other or has_school_only:
                return "course_location", None, False
            else:
                return "campuses_info", None, False

        # 5. follow_up_more_details
        elif follow_up_tag == "follow_up_more_details":
            if is_sca:
                bundle = ["sca_course_duration", "sca_eligibility", "sca_fees", "sca_location"]
                return "sca_info", bundle, False
            elif is_other:
                bundle = ["other_department_courses", "course_location", "other_department_fees"]
                return "other_department_courses", bundle, False
            elif has_school_only:
                if ctx.school_id == "computer_applications":
                    return "sca_info", None, False
                return "other_department_courses", None, False
            elif ctx.topic_intent:
                return ctx.topic_intent, None, False
            else:
                return "ask_which_course", None, False

        # 6. follow_up_other_course
        elif follow_up_tag == "follow_up_other_course":
            if ctx.topic_intent and new_course_id:
                # Re-run topic intent with new course
                return ctx.topic_intent, None, False
            elif ctx.topic_intent and not new_course_id:
                return "ask_which_course", None, False
            else:
                return "ask_what_to_know", None, False

        # Default fallback
        return follow_up_tag, None, False
