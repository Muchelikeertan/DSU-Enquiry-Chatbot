"""Unit tests for EntityExtractor longest-match handling and alias resolution."""

import json
import pytest
from app.config import settings
from app.nlp.entity_extractor import EntityExtractor


@pytest.fixture(scope="module")
def extractor():
    with open(settings.college_info_json_path, "r", encoding="utf-8") as f:
        college_data = json.load(f)
    return EntityExtractor(college_data=college_data)


def test_longest_match_bca_variants(extractor):
    """Ensure BCA in AI & DS is prioritized over generic BCA (longest match)."""
    # 1. Specialized BCA
    res_aids = extractor.extract("What is the fee for BCA in AI & DS?")
    assert res_aids.course_id == "sca_bca_ai_ds"
    assert res_aids.school_id == "computer_applications"

    res_aids2 = extractor.extract("Tell me about BCA in AI and Data Science")
    assert res_aids2.course_id == "sca_bca_ai_ds"

    # 2. General BCA
    res_bca = extractor.extract("What is the fee for regular BCA?")
    assert res_bca.course_id == "sca_bca"
    assert res_bca.school_id == "computer_applications"

    res_bca_full = extractor.extract("Bachelor of Computer Applications details")
    assert res_bca_full.course_id == "sca_bca"


def test_msc_and_bsc_variants(extractor):
    """Test Data Science and Cyber Security degrees in SCA."""
    res_msc_ds = extractor.extract("How much does M.Sc. Data Science cost?")
    assert res_msc_ds.course_id == "sca_msc_ds"
    assert res_msc_ds.school_id == "computer_applications"

    res_msc_cs = extractor.extract("Tell me about M.Sc in Cyber Security")
    assert res_msc_cs.course_id == "sca_msc_cs"
    assert res_msc_cs.school_id == "computer_applications"

    res_bsc_ds = extractor.extract("Eligibility for B.Sc. in Data Science")
    assert res_bsc_ds.course_id == "sca_bsc_ds"
    assert res_bsc_ds.school_id == "computer_applications"

    res_bsc_cs = extractor.extract("Tell me about B.Sc Cyber Security")
    assert res_bsc_cs.course_id == "sca_bsc_cs"
    assert res_bsc_cs.school_id == "computer_applications"


def test_mca_extraction(extractor):
    """Test MCA course resolution."""
    res_mca = extractor.extract("Is there an entrance exam for MCA?")
    assert res_mca.course_id == "sca_mca"
    assert res_mca.school_id == "computer_applications"


def test_non_sca_course_extraction(extractor):
    """Test extraction of Engineering, Management, and Law programmes."""
    res_btech = extractor.extract("What are the fees for B.Tech in Computer Science?")
    assert res_btech.course_id == "soe_btech"
    assert res_btech.school_id == "engineering"

    res_mba = extractor.extract("Tell me about the MBA course")
    assert res_mba.course_id == "scm_mba"
    assert res_mba.school_id == "commerce_management"

    res_law = extractor.extract("Admission for B.A. LL.B")
    assert res_law.course_id == "sol_ba_llb"
    assert res_law.school_id == "law"


def test_school_only_extraction(extractor):
    """Test extraction when only a school is named."""
    res_sca = extractor.extract("Tell me about the School of Computer Applications")
    assert res_sca.school_id == "computer_applications"
    assert res_sca.course_id is None

    res_soe = extractor.extract("Where is the School of Engineering located?")
    assert res_soe.school_id == "engineering"
    assert res_soe.course_id is None

    res_law = extractor.extract("Tell me about the Law school")
    assert res_law.school_id == "law"
    assert res_law.course_id is None


def test_non_existent_entity_never_invented(extractor):
    """Ensure absent courses or random words do not hallucinate an entity."""
    res = extractor.extract("Tell me about Astronomy and Astrophysics")
    assert res.course_id is None
    assert res.has_entities is False

    res2 = extractor.extract("What is the hostel fee?")
    assert res2.course_id is None
    assert res2.school_id is None
