"""Command-line interface to execute JSON knowledge integrity validation."""

import sys
from app.config import settings
from app.validator.json_validator import JSONIntegrityValidator


def main() -> int:
    """Executes the integrity validator and outputs formatted terminal summary."""
    # Reconfigure stdout for UTF-8 on Windows if supported
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=" * 80)
    print(" DSU College Enquiry Chatbot - Knowledge Base Integrity Validator")
    print("=" * 80)
    print(f"Intents:      {settings.intents_json_path}")
    print(f"College Info: {settings.college_info_json_path}")
    print(f"FAQ Info:     {settings.faq_info_json_path}")
    print("-" * 80)

    validator = JSONIntegrityValidator(
        intents_path=settings.intents_json_path,
        college_info_path=settings.college_info_json_path,
        faq_info_path=settings.faq_info_json_path,
    )

    result = validator.validate_all()

    if not result.syntax_ok:
        print("\n[CRITICAL ERROR] JSON Syntax or File Load Failure:")
        for err in result.errors:
            print(f"  [ERROR] {err}")
        return 1

    print("\nKnowledge Base Overview:")
    print(f"  - Registered Intents: {result.total_intents}")
    print(f"  - Curated FAQs:       {result.total_faqs}")
    print(f"  - Recognized Courses: {result.total_courses}")
    print(f"  - Recognized Schools: {result.total_schools}")
    print("\nIntegrity Rules Audit:")

    for r in result.rule_results:
        status_icon = "PASS" if r.passed else "FAIL"
        print(f"  [{status_icon:^6}] Rule {r.rule_number:02d}: {r.rule_name}")
        if not r.passed:
            for v in r.violations:
                print(f"           -> Violation: {v}")

    print("-" * 80)
    if result.is_valid:
        print("SUCCESS: All 10 integrity rules passed with 0 violations.")
        print("=" * 80)
        return 0
    else:
        print(f"FAILURE: Found {len(result.errors)} total integrity violation(s).")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
