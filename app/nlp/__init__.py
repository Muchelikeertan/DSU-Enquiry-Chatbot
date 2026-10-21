"""NLP Pipeline package for DSU College Enquiry Chatbot.

Provides:
- TextPreprocessor: normalization, acronym and degree expansions.
- IntentClassifier: TF-IDF (word + char_wb) and Logistic Regression classification.
- EntityExtractor: Longest-match greedy course, school, and specialisation extraction.
- FAQRetriever: Local TF-IDF cosine-similarity FAQ retrieval with form policy rules.
- ContextTracker: 6-turn TTL slot tracking and follow-up routing.
- NLPPipeline: Unified orchestrator for multi-turn NLP query processing.
"""

from app.nlp.classifier import IntentClassifier, IntentPrediction, ClassificationMetrics
from app.nlp.context_tracker import ContextTracker, ConversationSlotContext
from app.nlp.entity_extractor import EntityExtractor, ExtractedEntities
from app.nlp.faq_retriever import FAQRetriever, FAQMatchResult
from app.nlp.pipeline import NLPPipeline, ChatbotNLPResponse
from app.nlp.preprocessor import TextPreprocessor

__all__ = [
    "TextPreprocessor",
    "IntentClassifier",
    "IntentPrediction",
    "ClassificationMetrics",
    "EntityExtractor",
    "ExtractedEntities",
    "FAQRetriever",
    "FAQMatchResult",
    "ContextTracker",
    "ConversationSlotContext",
    "NLPPipeline",
    "ChatbotNLPResponse",
]
