"""Unit tests for TextPreprocessor edge cases, contractions, and domain variations."""

import pytest
from app.nlp.preprocessor import TextPreprocessor


@pytest.fixture
def preprocessor():
    return TextPreprocessor()


def test_basic_normalization(preprocessor):
    """Test lowercasing, whitespace stripping, and basic normalization."""
    assert preprocessor.normalize("  Hello World!  ") == "hello world"
    assert preprocessor.normalize("DAYANANDA   SAGAR") == "dayananda sagar"
    assert preprocessor.normalize("") == ""
    assert preprocessor.normalize("   ") == ""


def test_contraction_expansion(preprocessor):
    """Test standard English contractions."""
    assert "what is" in preprocessor.normalize("What's the fee?")
    assert "where is" in preprocessor.normalize("Where's the campus?")
    assert "it is" in preprocessor.normalize("It's located there")
    assert "cannot" in preprocessor.normalize("I can't find it")
    assert "do not" in preprocessor.normalize("Don't know")


def test_degree_abbreviation_standardization(preprocessor):
    """Test diverse spelling and punctuation variants of academic degrees."""
    assert preprocessor.normalize("B.Tech. in CSE") == "btech in cse"
    assert preprocessor.normalize("B.Tech in Aerospace") == "btech in aerospace"
    assert preprocessor.normalize("B Tech degree") == "btech degree"
    assert preprocessor.normalize("B.C.A. course") == "bca course"
    assert preprocessor.normalize("B.C.A") == "bca"
    assert preprocessor.normalize("M.C.A.") == "mca"
    assert preprocessor.normalize("B.Sc. Data Science") == "bsc data science"
    assert preprocessor.normalize("M.Sc. Cyber Security") == "msc cyber security"
    assert preprocessor.normalize("B.B.A.") == "bba"
    assert preprocessor.normalize("B.Com.") == "bcom"
    assert preprocessor.normalize("B.Pharm.") == "bpharm"
    assert preprocessor.normalize("Pharm.D.") == "pharmd"
    assert preprocessor.normalize("LL.B. (Hons.)") == "llb hons"
    assert preprocessor.normalize("LL.M.") == "llm"


def test_domain_acronym_standardization(preprocessor):
    """Test handling of domain acronyms like AI & DS, AI and DS, AIDS without semantic drift."""
    assert preprocessor.normalize("BCA AI & DS") == "bca ai and ds"
    assert preprocessor.normalize("BCA AI and DS") == "bca ai and ds"
    assert preprocessor.normalize("BCA AI-DS") == "bca ai and ds"
    assert preprocessor.normalize("BCA AI/DS") == "bca ai and ds"
    assert preprocessor.normalize("BCA AIDS") == "bca ai and ds"
    assert preprocessor.normalize("CSE AI & ML") == "cse ai and ml"
    assert preprocessor.normalize("Cybersecurity details") == "cyber security details"
    assert preprocessor.normalize("Cyber-Security details") == "cyber security details"
    assert preprocessor.normalize("Data Sciences program") == "data science program"


def test_punctuation_and_symbols(preprocessor):
    """Ensure non-alphanumeric noise is removed without losing word tokens."""
    assert preprocessor.normalize("Fee??? What is it?!?") == "fee what is it"
    assert preprocessor.normalize("Cost: Rs. 2,07,000/- per year") == "cost rs 2 07 000 per year"
    assert preprocessor.normalize("Email @ admission@dsu.edu.in") == "email admission dsu edu in"


def test_tokenization(preprocessor):
    """Ensure tokenize returns word list."""
    tokens = preprocessor.tokenize("What's the fee for B.Tech.?")
    assert tokens == ["what", "is", "the", "fee", "for", "btech"]
