"""Intent Classifier for DSU College Enquiry Chatbot.

Baseline Model: TF-IDF Vectorizer (word and char n-grams) + Logistic Regression.
Trained on all intent patterns in intents.json.
Supports confidence scoring, ambiguity detection, legacy tag alias resolution,
and stratified cross-validation evaluation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from app.nlp.preprocessor import TextPreprocessor


from sklearn.pipeline import Pipeline, FeatureUnion

@dataclass
class IntentPrediction:
    """Prediction outcome of the Intent Classifier."""
    tag: str
    confidence: float
    handler: str
    fallback_handler: Optional[str] = None
    requires_context: bool = False
    is_ambiguous: bool = False
    needs_fallback: bool = False
    secondary_tag: Optional[str] = None
    secondary_confidence: Optional[float] = None
    all_probabilities: Dict[str, float] = field(default_factory=dict)


@dataclass
class ClassificationMetrics:
    """Stratified cross-validation evaluation metrics."""
    accuracy: float
    macro_f1: float
    weighted_f1: float
    report: Dict[str, Any]
    confusion_matrix: List[List[int]]
    classes: List[str]
    n_samples: int
    n_classes: int


class IntentClassifier:
    """TF-IDF and Logistic Regression Intent Classifier."""

    def __init__(
        self,
        confidence_threshold: float = 0.60,
        ambiguity_margin: float = 0.08,
        preprocessor: Optional[TextPreprocessor] = None,
    ):
        self.confidence_threshold = confidence_threshold
        self.ambiguity_margin = ambiguity_margin
        self.preprocessor = preprocessor or TextPreprocessor()

        self.pipeline: Optional[Pipeline] = None
        self.classes_: List[str] = []
        self.intent_metadata: Dict[str, Dict[str, Any]] = {}
        self.legacy_aliases: Dict[str, str] = {}
        self.is_trained: bool = False

    def _extract_dataset(self, intents_data: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Extracts and normalizes (text, label) pairs from intents.json patterns."""
        texts: List[str] = []
        labels: List[str] = []

        self.legacy_aliases = intents_data.get("legacy_tag_aliases", {})

        for intent in intents_data.get("intents", []):
            tag = intent.get("tag")
            if not tag:
                continue

            self.intent_metadata[tag] = {
                "handler": intent.get("handler", "local"),
                "fallback_handler": intent.get("fallback_handler"),
                "requires_context": intent.get("requires_context", False),
                "responses": intent.get("responses", []),
            }

            for pattern in intent.get("patterns", []):
                norm_pattern = self.preprocessor.normalize(pattern)
                if norm_pattern:
                    texts.append(norm_pattern)
                    labels.append(tag)

        return texts, labels

    def _build_pipeline(self) -> Pipeline:
        """Constructs a robust FeatureUnion (word + char_wb) and LogisticRegression pipeline."""
        union_features = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, min_df=1)),
        ])

        return Pipeline([
            ("features", union_features),
            ("clf", LogisticRegression(
                C=5.0,
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            )),
        ])

    def fit(self, intents_data: Dict[str, Any]) -> "IntentClassifier":
        """Trains the TF-IDF and Logistic Regression pipeline on the intents patterns."""
        texts, labels = self._extract_dataset(intents_data)

        if not texts:
            raise ValueError("No training patterns found in provided intents_data.")

        self.pipeline = self._build_pipeline()
        self.pipeline.fit(texts, labels)
        self.classes_ = list(self.pipeline.named_steps["clf"].classes_)
        self.is_trained = True
        return self

    def predict(self, text: str) -> IntentPrediction:
        """Classifies a user message with confidence and ambiguity checks."""
        if not self.is_trained or self.pipeline is None:
            raise RuntimeError("IntentClassifier has not been trained. Call fit() first.")

        norm_text = self.preprocessor.normalize(text)
        if not norm_text:
            # Handle empty query gracefully
            return IntentPrediction(
                tag="fallback",
                confidence=0.0,
                handler="llm",
                needs_fallback=True,
            )

        # Compute class probabilities
        probs = self.pipeline.predict_proba([norm_text])[0]
        sorted_indices = np.argsort(probs)[::-1]

        top_idx = sorted_indices[0]
        top_tag = self.classes_[top_idx]
        top_prob = float(probs[top_idx])

        # Resolve legacy tag aliases if any
        resolved_tag = self.legacy_aliases.get(top_tag, top_tag)

        # Check secondary candidate for ambiguity
        secondary_tag = None
        secondary_prob = None
        is_ambiguous = False

        if len(sorted_indices) > 1:
            sec_idx = sorted_indices[1]
            secondary_tag = self.classes_[sec_idx]
            secondary_prob = float(probs[sec_idx])
            # If difference between top 1 and top 2 is small and top confidence is moderate
            if (top_prob - secondary_prob) < self.ambiguity_margin and top_prob < 0.70:
                is_ambiguous = True

        # Fallback check
        needs_fallback = top_prob < self.confidence_threshold

        meta = self.intent_metadata.get(resolved_tag, {})
        handler = meta.get("handler", "local")
        fallback_handler = meta.get("fallback_handler")
        requires_context = meta.get("requires_context", False)

        prob_dict = {self.classes_[i]: float(probs[i]) for i in sorted_indices[:5]}

        return IntentPrediction(
            tag=resolved_tag,
            confidence=round(top_prob, 4),
            handler=handler,
            fallback_handler=fallback_handler,
            requires_context=requires_context,
            is_ambiguous=is_ambiguous,
            needs_fallback=needs_fallback,
            secondary_tag=secondary_tag,
            secondary_confidence=round(secondary_prob, 4) if secondary_prob is not None else None,
            all_probabilities=prob_dict,
        )

    def evaluate(self, intents_data: Dict[str, Any], n_splits: int = 3) -> ClassificationMetrics:
        """Performs stratified k-fold cross-validation without data leakage."""
        texts, labels = self._extract_dataset(intents_data)

        # Filter out classes that have fewer samples than n_splits
        label_counts = {lbl: labels.count(lbl) for lbl in set(labels)}
        valid_indices = [idx for idx, lbl in enumerate(labels) if label_counts[lbl] >= n_splits]

        X_filtered = [texts[i] for i in valid_indices]
        y_filtered = [labels[i] for i in valid_indices]

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        all_y_true: List[str] = []
        all_y_pred: List[str] = []

        for train_idx, test_idx in skf.split(X_filtered, y_filtered):
            X_train = [X_filtered[i] for i in train_idx]
            y_train = [y_filtered[i] for i in train_idx]
            X_test = [X_filtered[i] for i in test_idx]
            y_test = [y_filtered[i] for i in test_idx]

            fold_pipeline = self._build_pipeline()

            fold_pipeline.fit(X_train, y_train)
            preds = fold_pipeline.predict(X_test)

            all_y_true.extend(y_test)
            all_y_pred.extend(preds)

        unique_classes = sorted(list(set(all_y_true)))
        acc = accuracy_score(all_y_true, all_y_pred)
        macro_f1 = f1_score(all_y_true, all_y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(all_y_true, all_y_pred, average="weighted", zero_division=0)
        report = classification_report(all_y_true, all_y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(all_y_true, all_y_pred, labels=unique_classes).tolist()

        return ClassificationMetrics(
            accuracy=round(float(acc), 4),
            macro_f1=round(float(macro_f1), 4),
            weighted_f1=round(float(weighted_f1), 4),
            report=report,
            confusion_matrix=cm,
            classes=unique_classes,
            n_samples=len(X_filtered),
            n_classes=len(unique_classes),
        )
