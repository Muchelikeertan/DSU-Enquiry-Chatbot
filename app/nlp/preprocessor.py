"""Text Preprocessor for DSU College Enquiry Chatbot.

Provides text normalization, case folding, contraction expansion,
domain-specific acronym standardization ('AI & DS' -> 'ai and ds', 'B.Tech.' -> 'btech'),
and tokenization.
"""

import re
import unicodedata
from typing import List


class TextPreprocessor:
    """Normalizes and tokenizes user queries for intent classification and FAQ retrieval."""

    def __init__(self, expand_acronyms: bool = True):
        self.expand_acronyms = expand_acronyms

        # Contraction mapping
        self.contractions = {
            r"\bwhat's\b": "what is",
            r"\bwhere's\b": "where is",
            r"\bhow's\b": "how is",
            r"\bwho's\b": "who is",
            r"\bthere's\b": "there is",
            r"\bit's\b": "it is",
            r"\bthat's\b": "that is",
            r"\bcan't\b": "cannot",
            r"\bwon't\b": "will not",
            r"\bdon't\b": "do not",
            r"\bdoesn't\b": "does not",
            r"\bdidn't\b": "did not",
            r"\bi'm\b": "i am",
            r"\byou're\b": "you are",
            r"\bwe're\b": "we are",
            r"\bthey're\b": "they are",
        }

        # Academic degree standardization (regex -> standard token)
        self.degree_patterns = [
            (re.compile(r"\bb\.?\s*tech\.?\b", re.IGNORECASE), "btech"),
            (re.compile(r"\bm\.?\s*tech\.?\b", re.IGNORECASE), "mtech"),
            (re.compile(r"\bb\.?\s*c\.?\s*a\.?\b", re.IGNORECASE), "bca"),
            (re.compile(r"\bm\.?\s*c\.?\s*a\.?\b", re.IGNORECASE), "mca"),
            (re.compile(r"\bb\.?\s*sc\.?\b", re.IGNORECASE), "bsc"),
            (re.compile(r"\bm\.?\s*sc\.?\b", re.IGNORECASE), "msc"),
            (re.compile(r"\bb\.?\s*b\.?\s*a\.?\b", re.IGNORECASE), "bba"),
            (re.compile(r"\bb\.?\s*com\.?\b", re.IGNORECASE), "bcom"),
            (re.compile(r"\bb\.?\s*pharm\.?\b", re.IGNORECASE), "bpharm"),
            (re.compile(r"\bm\.?\s*pharm\.?\b", re.IGNORECASE), "mpharm"),
            (re.compile(r"\bpharm\.?\s*d\.?\b", re.IGNORECASE), "pharmd"),
            (re.compile(r"\bb\.?\s*p\.?\s*t\.?\b", re.IGNORECASE), "bpt"),
            (re.compile(r"\bm\.?\s*p\.?\s*t\.?\b", re.IGNORECASE), "mpt"),
            (re.compile(r"\bb\.?\s*des\.?\b", re.IGNORECASE), "bdes"),
            (re.compile(r"\bb\.?\s*a\.?\b", re.IGNORECASE), "ba"),
            (re.compile(r"\bll\.?\s*b\.?\b", re.IGNORECASE), "llb"),
            (re.compile(r"\bll\.?\s*m\.?\b", re.IGNORECASE), "llm"),
            (re.compile(r"\bph\.?\s*d\.?\b", re.IGNORECASE), "phd"),
        ]

        # Domain term variations and common typos/synonyms
        self.domain_synonyms = [
            # Ampersand conversion
            (re.compile(r"&"), " and "),
            # Conversational shorthand
            (re.compile(r"\bwht\b", re.IGNORECASE), "what"),
            (re.compile(r"\b(?:u|ya)\b", re.IGNORECASE), "you"),
            (re.compile(r"\br\b", re.IGNORECASE), "are"),
            (re.compile(r"\bplz\b", re.IGNORECASE), "please"),
            (re.compile(r"\btell\s+me\s+abt\b", re.IGNORECASE), "tell me about"),
            (re.compile(r"\babt\b", re.IGNORECASE), "about"),
            # Common typos
            (re.compile(r"\b(?:scince|scienc|sceince)\b", re.IGNORECASE), "science"),
            (re.compile(r"\b(?:eligiblity|eligibilty|eligiblty|elgibility)\b", re.IGNORECASE), "eligibility"),
            (re.compile(r"\b(?:anuel|anuall|annuel|anual)\b", re.IGNORECASE), "annual"),
            (re.compile(r"\b(?:hostle|hostal)\b", re.IGNORECASE), "hostel"),
            (re.compile(r"\b(?:programe|programme)\b", re.IGNORECASE), "program"),
            (re.compile(r"\b(?:clg|colg|collg)\b", re.IGNORECASE), "college"),
            (re.compile(r"\b(?:admisn|admisison|admision)\b", re.IGNORECASE), "admission"),
            # AI & DS / AI and DS / AI-DS / AIDS
            (re.compile(r"\bai\s*(?:and|&|\/|\-)\s*ds\b", re.IGNORECASE), "ai and ds"),
            (re.compile(r"\baids\b", re.IGNORECASE), "ai and ds"),
            # AI & ML / AI and ML
            (re.compile(r"\bai\s*(?:and|&|\/|\-)\s*ml\b", re.IGNORECASE), "ai and ml"),
            # Cybersecurity / Cyber Security
            (re.compile(r"\bcybersecurity\b", re.IGNORECASE), "cyber security"),
            (re.compile(r"\bcyber\-security\b", re.IGNORECASE), "cyber security"),
            # Data Sciences -> Data Science
            (re.compile(r"\bdata\s+sciences\b", re.IGNORECASE), "data science"),
            # Fees / Fee
            (re.compile(r"\bfees\b", re.IGNORECASE), "fee"),
            (re.compile(r"\bcosts\b", re.IGNORECASE), "cost"),
        ]

    def normalize(self, text: str) -> str:
        """Applies full normalization pipeline to user query text.

        1. Unicode normalization (NFKD) and case folding.
        2. Contraction expansion.
        3. Academic degree and domain acronym standardization.
        4. Punctuation removal (preserving alphanumeric and word spacing).
        5. Whitespace consolidation.
        """
        if not text:
            return ""

        # 1. Unicode decomposition & lowercase
        normalized = unicodedata.normalize("NFKD", text).strip().lower()

        # 2. Expand contractions
        for pattern, replacement in self.contractions.items():
            normalized = re.sub(pattern, replacement, normalized)

        # 3. Standardize degrees and domain terms if enabled
        if self.expand_acronyms:
            for pattern, replacement in self.degree_patterns:
                normalized = pattern.sub(replacement, normalized)

            for pattern, replacement in self.domain_synonyms:
                normalized = pattern.sub(replacement, normalized)

        # 4. Remove unwanted punctuation (keep letters, numbers, spaces)
        normalized = re.sub(r"[^\w\s]", " ", normalized)

        # 5. Collapse multiple whitespaces
        normalized = re.sub(r"\s+", " ", normalized).strip()

        return normalized

    def tokenize(self, text: str) -> List[str]:
        """Normalizes and splits text into whitespace-delimited word tokens."""
        clean_text = self.normalize(text)
        if not clean_text:
            return []
        return clean_text.split()
