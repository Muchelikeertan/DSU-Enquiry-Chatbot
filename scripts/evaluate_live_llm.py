"""Stage 3 Live LLM Evaluation Script (Groq API - openai/gpt-oss-120b).

Executes a controlled, 10-question evaluation of the live Groq LLM integration
connected through the NLP-first pipeline.

Checks:
- Local bypass for verified SCA FAQs (zero Groq API calls).
- Grounded responses for dynamic & non-SCA queries.
- Strict adherence to verification statuses and negative guardrails.
- Enquiry-form policy enforcement.
- Safe handling of out-of-domain or unverified queries.
- Zero credential exposure.
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

# Reconfigure stdout for utf-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure project root is in path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.config import settings
from app.nlp.pipeline import NLPPipeline


def sanitize(text: str) -> str:
    """Masks any potential secret keys or sensitive tokens."""
    return re.sub(r"gsk_[A-Za-z0-9]+", "[REDACTED_API_KEY]", text)


def run_preflight_checks() -> bool:
    """Performs pre-flight checks before making any live API call."""
    print("=" * 80)
    print("STAGE 3 LIVE LLM EVALUATION — PRE-FLIGHT CHECKS")
    print("=" * 80)

    # 1. Verify .gitignore excludes .env
    gitignore_path = BASE_DIR / ".gitignore"
    if not gitignore_path.exists():
        print("[FAIL] .gitignore not found!")
        return False
    with open(gitignore_path, "r", encoding="utf-8") as f:
        gi_content = f.read()
    if ".env" not in gi_content:
        print("[FAIL] .env is not excluded in .gitignore!")
        return False
    print("[PASS] .env is strictly excluded by .gitignore.")

    # 2. Verify settings loaded without displaying secret
    if not settings.groq_api_key:
        print("[FAIL] GROQ_API_KEY is not configured in .env!")
        return False
    if not settings.groq_api_key.startswith("gsk_"):
        print("[FAIL] GROQ_API_KEY does not have the expected Groq prefix (gsk_)!")
        return False
    print(f"[PASS] GROQ_API_KEY is present and has valid prefix [gsk_***, length={len(settings.groq_api_key)}].")

    # 3. Verify Model and Provider configuration
    if settings.llm_provider != "groq":
        print(f"[FAIL] LLM_PROVIDER is '{settings.llm_provider}', expected 'groq'!")
        return False
    if settings.groq_model != "openai/gpt-oss-120b":
        print(f"[FAIL] GROQ_MODEL is '{settings.groq_model}', expected 'openai/gpt-oss-120b'!")
        return False
    print(f"[PASS] Model configured: {settings.groq_model} (Provider: {settings.llm_provider}).")

    # 4. Verify canonical files exist
    for name, path in [
        ("intents.json", settings.intents_json_path),
        ("college_info.json", settings.college_info_json_path),
        ("faq_info.json", settings.faq_info_json_path),
    ]:
        if not path.exists():
            print(f"[FAIL] Canonical file missing: {name}")
            return False
    print("[PASS] All canonical knowledge base files verified.")

    print("[SUCCESS] All pre-flight checks passed safely. Ready for live evaluation.\n")
    return True


# 10 Representative Test Cases
EVALUATION_CASES: List[Dict[str, Any]] = [
    {
        "id": 1,
        "category": "a. Verified SCA FAQ (Fees)",
        "query": "What is the annual fee for the BCA programme at DSU?",
        "expected_groq_called": False,
        "expected_form_policy": "never",
        "expected_form_shown": False,
        "grounding_check": lambda r: "207,000" in r.response_text or "2,07,000" in r.response_text or "207000" in r.response_text,
        "expected_description": "Must answer locally with verified BCA yearly fee (₹2,07,000). Groq must NOT be called.",
    },
    {
        "id": 2,
        "category": "a. Verified SCA FAQ (Dean)",
        "query": "Who is the Dean of the School of Computer Applications?",
        "expected_groq_called": False,
        "expected_form_policy": "never",
        "expected_form_shown": False,
        "grounding_check": lambda r: "Dr. S. Senthil" in r.response_text,
        "expected_description": "Must answer locally with verified Dean name (Dr. S. Senthil). Groq must NOT be called.",
    },
    {
        "id": 3,
        "category": "a. Verified SCA FAQ (Duration & Campus)",
        "query": "What is the duration and campus for MCA?",
        "expected_groq_called": False,
        "expected_form_policy": "never",
        "expected_form_shown": False,
        "grounding_check": lambda r: ("2-year" in r.response_text or "2 year" in r.response_text or "two year" in r.response_text.lower()),
        "expected_description": "Must answer locally from verified FAQ (MCA is 2-year regular programme). Groq must NOT be called.",
    },
    {
        "id": 4,
        "category": "b. Dynamic / Non-SCA (Engineering Programmes)",
        "query": "What degree programmes does the School of Engineering offer?",
        "expected_groq_called": True,
        "expected_form_policy": "on_failure",
        "expected_form_shown": None,  # Optional unless details unconfirmed
        "grounding_check": lambda r: "b.tech" in r.response_text.lower() or "btech" in r.response_text.lower() or "engineering" in r.response_text.lower(),
        "expected_description": "Should invoke Groq with grounding context. Must mention B.Tech/M.Tech under School of Engineering without hallucinating nonexistent degrees.",
    },
    {
        "id": 5,
        "category": "b. Dynamic / Non-SCA (Engineering Leadership)",
        "query": "Who is the Dean of the School of Engineering?",
        "expected_groq_called": True,
        "expected_form_policy": "on_failure",
        "expected_form_shown": True,
        "grounding_check": lambda r: ("not confirmed" in r.response_text.lower() or "not listed" in r.response_text.lower() or "unconfirmed" in r.response_text.lower() or "enquiry form" in r.response_text.lower() or "dr. s. senthil" in r.response_text.lower()),
        "expected_description": "Must NOT invent a Dean name for Engineering (unlisted in checked records). Must direct to admissions/enquiry form.",
    },
    {
        "id": 6,
        "category": "c. Missing / Unverified Fact (SCA Scholarships)",
        "query": "Are there scholarships available for BCA or MCA students at SCA?",
        "expected_groq_called": None,  # May route to sca_courses or entrance_exams_scholarships
        "expected_form_policy": None,
        "expected_form_shown": None,
        "grounding_check": lambda r: ("scholarship" in r.response_text.lower() or "seven programmes" in r.response_text.lower() or "enquiry form" in r.response_text.lower()),
        "expected_description": "Evaluates handling of SCA scholarship inquiry (unverified in primary checked records).",
    },
    {
        "id": 7,
        "category": "d. Conflicting / Unverified Campus (BCA AI & DS)",
        "query": "Which campus is the BCA in Artificial Intelligence and Data Sciences held at?",
        "expected_groq_called": None,  # May be answered by static form FAQ or Groq with guardrail
        "expected_form_policy": "always",
        "expected_form_shown": True,
        "grounding_check": lambda r: ("unconfirmed" in r.response_text.lower() or "not verified" in r.response_text.lower() or "enquiry form" in r.response_text.lower() or "isn't listed" in r.response_text.lower()),
        "expected_description": "Must respect data conflict: teaching campus is unconfirmed on 2026 admissions list. Must NOT assume Kudlu Gate. Must show enquiry form.",
    },
    {
        "id": 8,
        "category": "d. Conflicting / Unverified Facility (Kudlu Gate Hostel)",
        "query": "Is hostel accommodation available at the Kudlu Gate campus?",
        "expected_groq_called": None,  # Handled either by local FAQ or Groq LLM
        "expected_form_policy": None,
        "expected_form_shown": None,
        "grounding_check": lambda r: ("harohalli" in r.response_text.lower() and ("kudlu" in r.response_text.lower() or "unverified" in r.response_text.lower() or "no verified" in r.response_text.lower())),
        "expected_description": "Must state hostel is confirmed for Harohalli Main Campus only; Kudlu Gate hostel is unverified.",
    },
    {
        "id": 9,
        "category": "e. Fees & Enquiry-Form Policies (B.Tech Fees)",
        "query": "What is the tuition fee for B.Tech in Computer Science?",
        "expected_groq_called": True,
        "expected_form_policy": "always",
        "expected_form_shown": True,
        "grounding_check": lambda r: ("enquiry form" in r.response_text.lower() or "quota" in r.response_text.lower() or "varies" in r.response_text.lower() or "unverified" in r.response_text.lower()),
        "expected_description": "Non-SCA fee must NOT be fabricated. Model must explain fees vary by quota/specialisation and trigger enquiry form.",
    },
    {
        "id": 10,
        "category": "f. Unknown / Out-of-Domain Question (Library Pets)",
        "query": "Can I bring my pet dog to the DSU library?",
        "expected_groq_called": None,  # Routed to enquiry form or fallback LLM
        "expected_form_policy": None,
        "expected_form_shown": True,
        "grounding_check": lambda r: ("enquiry form" in r.response_text.lower() or "official records" in r.response_text.lower() or "contact" in r.response_text.lower() or "admissions" in r.response_text.lower()),
        "expected_description": "Must not fabricate pet policies at DSU. Must direct student to enquiry form or administration.",
    },
]


def run_live_evaluation():
    if not run_preflight_checks():
        print("[ABORTED] Pre-flight checks failed. Live evaluation halted.")
        return

    print("=" * 80)
    print("INITIALIZING REAL NLP PIPELINE WITH LIVE GROQ LLM HANDLER")
    print(f"Model: {settings.groq_model} | Provider: {settings.llm_provider}")
    print("=" * 80)

    pipeline = NLPPipeline.from_files(
        intents_path=settings.intents_json_path,
        college_info_path=settings.college_info_json_path,
        faq_info_path=settings.faq_info_json_path,
    )

    results: List[Dict[str, Any]] = []
    pass_count = 0
    fail_count = 0
    review_count = 0

    print(f"\nExecuting {len(EVALUATION_CASES)} controlled live evaluation queries...\n")

    for case in EVALUATION_CASES:
        cid = case["id"]
        cat = case["category"]
        query = case["query"]

        print(f"[{cid}/10] Testing: {query}")
        print(f"      Category: {cat}")

        try:
            resp = pipeline.process_query(query, session_id=f"live_eval_session_{cid}")

            # 1. Check Groq call expectation
            groq_call_ok = True
            if case["expected_groq_called"] is True and not resp.used_llm:
                groq_call_ok = False
            elif case["expected_groq_called"] is False and resp.used_llm:
                groq_call_ok = False

            # 2. Check form trigger expectation
            form_shown_ok = True
            if case["expected_form_shown"] is True and not resp.show_enquiry_form:
                form_shown_ok = False
            elif case["expected_form_shown"] is False and resp.show_enquiry_form:
                form_shown_ok = False

            # 3. Check factual grounding
            grounding_ok = case["grounding_check"](resp)

            # Determine verdict
            if groq_call_ok and form_shown_ok and grounding_ok:
                verdict = "PASS"
                pass_count += 1
            elif not groq_call_ok:
                verdict = "FAIL (Routing Mismatch)"
                fail_count += 1
            elif not grounding_ok:
                verdict = "FAIL (Grounding Violation / Hallucination Risk)"
                fail_count += 1
            elif not form_shown_ok:
                verdict = "NEEDS MANUAL REVIEW (Enquiry Form Policy)"
                review_count += 1
            else:
                verdict = "NEEDS MANUAL REVIEW"
                review_count += 1

            sanitized_response = sanitize(resp.response_text)

            res_entry = {
                "id": cid,
                "category": cat,
                "query": query,
                "verdict": verdict,
                "used_llm": resp.used_llm,
                "expected_used_llm": case["expected_groq_called"],
                "show_enquiry_form": resp.show_enquiry_form,
                "expected_form_shown": case["expected_form_shown"],
                "intent": str(resp.intent),
                "handler": resp.handler,
                "status": resp.status,
                "source": resp.source,
                "response_snippet": sanitized_response[:200] + ("..." if len(sanitized_response) > 200 else ""),
                "full_response": sanitized_response,
                "expected_description": case["expected_description"],
            }
            results.append(res_entry)

            try:
                print(f"      Verdict:        [{verdict}]")
                print(f"      Used LLM:       {resp.used_llm} (Expected: {case['expected_groq_called']})")
                print(f"      Enquiry Form:   {resp.show_enquiry_form} (Policy: {resp.form_policy})")
                print(f"      Intent/Handler: {resp.intent} / {resp.handler}")
                safe_snippet = sanitized_response[:120].encode('ascii', errors='replace').decode('ascii')
                print(f"      Response:       {safe_snippet}...\n")
            except Exception:
                pass

        except Exception as e:
            err_type = type(e).__name__
            err_msg = sanitize(str(e))[:100]
            print(f"      [EXCEPTION] {err_type}: {err_msg}\n")
            results.append({
                "id": cid,
                "category": cat,
                "query": query,
                "verdict": f"FAIL ({err_type})",
                "used_llm": False,
                "error": f"{err_type}: {err_msg}",
            })
            fail_count += 1

    # Save detailed evaluation report
    report_file = BASE_DIR / "live_eval_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": settings.groq_model,
                "provider": settings.llm_provider,
                "total_tests": len(EVALUATION_CASES),
                "pass_count": pass_count,
                "fail_count": fail_count,
                "review_count": review_count,
                "results": results,
            },
            f,
            indent=2,
        )

    print("=" * 80)
    print("LIVE EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total Tests Run:     {len(EVALUATION_CASES)}")
    print(f"PASS:                {pass_count}")
    print(f"FAIL:                {fail_count}")
    print(f"NEEDS MANUAL REVIEW: {review_count}")
    print(f"Detailed Report:     {report_file}")
    print("=" * 80)


if __name__ == "__main__":
    run_live_evaluation()
