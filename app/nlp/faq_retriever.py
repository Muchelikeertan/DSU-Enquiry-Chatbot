"""Local FAQ Retrieval Engine for DSU College Enquiry Chatbot.

Retrieves verified answers locally using TF-IDF vectorization and cosine similarity
over the 80 curated FAQs and their 369 question variations in faq_info.json.
Respects intent filtering, course entity mapping, verification statuses,
and enquiry form policies (never, always, on_failure).
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.nlp.preprocessor import TextPreprocessor


from sklearn.pipeline import FeatureUnion

@dataclass
class FAQMatchResult:
    """Outcome of local FAQ retrieval."""
    faq_id: str
    intent: str
    score: float
    answer: Optional[str]
    answer_mode: str  # 'static' | 'dynamic' | 'form'
    form_policy: str  # 'never' | 'always' | 'on_failure'
    show_enquiry_form: bool
    status: str
    source: Optional[str] = None
    course_id: Optional[str] = None
    is_confident: bool = True
    matched_question: Optional[str] = None


class FAQRetriever:
    """Local vector-based FAQ retrieval engine."""

    def __init__(
        self,
        confidence_threshold: float = 0.65,
        scoped_threshold: float = 0.20,
        preprocessor: Optional[TextPreprocessor] = None,
        faq_data: Optional[Dict[str, Any]] = None,
    ):
        self.confidence_threshold = confidence_threshold
        self.scoped_threshold = scoped_threshold
        self.preprocessor = preprocessor or TextPreprocessor()

        self.faqs: List[Dict[str, Any]] = []
        self.faq_id_map: Dict[str, Dict[str, Any]] = {}
        # List of (faq_index, normalized_question_text, raw_question_text)
        self.indexed_questions: List[Tuple[int, str, str]] = []
        self.vectorizer: Optional[Any] = None
        self.question_vectors: Optional[Any] = None

        if faq_data:
            self.index_faqs(faq_data)

    def index_faqs(self, faq_data: Dict[str, Any]) -> None:
        """Builds TF-IDF index over all questions across all 80 FAQs."""
        self.faqs = faq_data.get("faqs", [])
        self.faq_id_map = {f["id"]: f for f in self.faqs if "id" in f}
        self.indexed_questions = []

        all_norm_questions: List[str] = []

        for idx, faq in enumerate(self.faqs):
            questions = faq.get("questions", [])
            for q in questions:
                norm_q = self.preprocessor.normalize(q)
                if norm_q:
                    self.indexed_questions.append((idx, norm_q, q))
                    all_norm_questions.append(norm_q)

        if not all_norm_questions:
            return

        self.vectorizer = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, min_df=1)),
        ])
        self.question_vectors = self.vectorizer.fit_transform(all_norm_questions)

    def retrieve(
        self,
        query: str,
        predicted_intent: Optional[str] = None,
        course_id: Optional[str] = None,
        school_id: Optional[str] = None,
    ) -> Optional[FAQMatchResult]:
        """Finds the most relevant FAQ using scoped TF-IDF cosine similarity.

        Filtering priority:
        1. Both intent AND course_id match.
        2. Course_id matches.
        3. Intent matches.
        4. Global FAQ search.
        """
        if not self.vectorizer or self.question_vectors is None or not self.indexed_questions:
            return None

        norm_query = self.preprocessor.normalize(query)
        if not norm_query:
            return None

        query_vec = self.vectorizer.transform([norm_query])
        similarities = cosine_similarity(query_vec, self.question_vectors)[0]

        # Determine candidate question indices based on scoping
        candidate_indices: List[int] = []

        is_scoped = False
        # 1. Filter: Intent + Course ID
        if predicted_intent and course_id:
            for q_idx, (faq_idx, _, _) in enumerate(self.indexed_questions):
                f = self.faqs[faq_idx]
                f_cids = [f.get("course_id")] + f.get("course_ids", [])
                if f.get("intent") == predicted_intent and course_id in f_cids:
                    candidate_indices.append(q_idx)
            if candidate_indices:
                is_scoped = True

        # 2. Filter: Course ID only (if no matches on intent+course)
        if not candidate_indices and course_id:
            for q_idx, (faq_idx, _, _) in enumerate(self.indexed_questions):
                f = self.faqs[faq_idx]
                f_cids = [f.get("course_id")] + f.get("course_ids", [])
                if course_id in f_cids:
                    candidate_indices.append(q_idx)

        # 3. Filter: Intent only (if no course_id or no matches)
        if not candidate_indices and predicted_intent:
            for q_idx, (faq_idx, _, _) in enumerate(self.indexed_questions):
                f = self.faqs[faq_idx]
                if f.get("intent") == predicted_intent:
                    candidate_indices.append(q_idx)

        # 4. Fallback: Search all questions if still no candidates
        if not candidate_indices:
            candidate_indices = list(range(len(self.indexed_questions)))

        # Find best scoring candidate
        best_q_idx = candidate_indices[0]
        best_score = float(similarities[best_q_idx])

        for q_idx in candidate_indices[1:]:
            score = float(similarities[q_idx])
            if score > best_score:
                best_score = score
                best_q_idx = q_idx

        faq_idx, _, raw_q = self.indexed_questions[best_q_idx]
        matched_faq = self.faqs[faq_idx]

        threshold = self.scoped_threshold if is_scoped else self.confidence_threshold
        is_confident = best_score >= threshold
        form_policy = matched_faq.get("form_policy", "never")

        # Determine show_enquiry_form according to intents.json enquiry_form_rules
        if form_policy == "always":
            show_enquiry_form = True
        elif form_policy == "on_failure":
            show_enquiry_form = not is_confident
        else:  # never
            show_enquiry_form = False

        return FAQMatchResult(
            faq_id=matched_faq["id"],
            intent=matched_faq.get("intent", predicted_intent or "unknown"),
            score=round(best_score, 4),
            answer=matched_faq.get("answer"),
            answer_mode=matched_faq.get("answer_mode", "static"),
            form_policy=form_policy,
            show_enquiry_form=show_enquiry_form,
            status=matched_faq.get("status", "source_checked"),
            source=matched_faq.get("source"),
            course_id=matched_faq.get("course_id"),
            is_confident=is_confident,
            matched_question=raw_q,
        )
