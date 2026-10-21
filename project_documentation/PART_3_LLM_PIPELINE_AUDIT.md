# PART 3: COMPLETE LLM PIPELINE AUDIT
## Grounded LLM Fallback Architecture & Guardrails

**Codebase Implementation:** `app/llm/handler.py`, `app/retrieval/dsu_retriever.py`, `app/nlp/pipeline.py`  
**Provider & SDK:** Groq Cloud Platform (`groq` Python SDK v0.9+)  
**Configured Model:** `openai/gpt-oss-120b` (Default in `handler.py`, configurable to `llama-3.3-70b-versatile` via `.env`)  

---

## 1. Architectural Role of the LLM

In this project, **the LLM is NOT an autonomous conversational agent, and it is NOT used for primary intent classification.**

Instead, the system implements a **Deterministic-First, LLM-Assisted Hybrid Architecture**:
1. **Primary Layer:** Fast, deterministic local NLP (TF-IDF + Logistic Regression + Cosine FAQ Search).
2. **Fallback Layer:** The LLM is invoked **only** under strict, rule-governed conditions where the local knowledge base is insufficient or unverified.

### Permitted LLM Triggers (`handler.py:L39`):
An LLM call is authorized **only** if one of the following criteria is met:
1. **Low Classifier Confidence:** Intent classifier confidence falls below `0.60`.
2. **Ambiguous Query:** Difference between the top two intent probabilities is $\le 0.08$.
3. **Intent Tag Explicitly Delegates to LLM:** Intents mapped to handlers:
   - `handler == "llm"`: e.g., `other_department_courses`, `other_department_faculty`, `course_location`, `fallback`.
   - `handler == "llm_fee_lookup"`: e.g., `other_department_fees` (inquiries regarding Engineering, Law, Pharmacy, or Management fees).
   - `handler == "local_then_llm"`: When local FAQ cosine similarity is below `0.65`.

---

## 2. Dynamic Retrieval & Grounding Context Assembly (`app/retrieval/dsu_retriever.py`)

Before calling the Groq API, the system collects verified ground-truth context using `DSURetriever`.

### How Evidence is Retrieved:
1. **Catalog Matching:** Checks `college_info.json` for verified university data (accreditation, campuses, placement records, recruiters, SCA programs).
2. **Keyword Relevance Scoring:** Matches user query tokens against known DSU departmental and admissions documentation.
3. **Structured Packaging:** Assembles verified facts, program names, and official source URLs into a formatted context block:
   ```text
   GROUNDING CONTEXT:
   - Institution: Dayananda Sagar University (DSU), Bangalore
   - Approved By: UGC, AICTE, PCI, BCI, INC
   - Campuses: Harohalli (Main Campus, Kanakapura Road) & Kudlu Gate (City Campus)
   - Highest Placement Package: 32 LPA; Top Recruiters: Amazon, Cognizant, Infosys, IBM
   ...
   OFFICIAL SOURCES:
   - [DSU Official Admissions](https://www.dsu.edu.in/admissions)
   - [DSU Fee Structure](https://www.dsu.edu.in/eligibility)
   ```

---

## 3. Strict Anti-Hallucination Guardrails (`app/llm/handler.py:L41-64`)

The Groq API is constrained by an enforceable `SYSTEM_PROMPT` containing seven strict rules:

| Rule Number | Guardrail Name | Technical Constraint Enforced |
| :--- | :--- | :--- |
| **Rule 1** | **Direct Answer Rule** | If the information exists in the Grounding Context, the model must answer directly with exact numbers, packages, and criteria. |
| **Rule 2** | **Complete Sentences** | Responses must never end abruptly or in fragmented sentences. |
| **Rule 3** | **No Limitation Disclaimers** | The model is forbidden from saying *"I am just an AI"* or *"The available webpages do not contain this answer"*. |
| **Rule 4** | **Critical Fee-Category Protection** | **Crucial Academic Rule:** A general tuition fee can NEVER be cited for Management Quota, COMED-K, CET, Hostel, or Transport fees. If unverified, it MUST direct to Admissions. |
| **Rule 5** | **Multi-Part Tracking** | In compound questions (e.g., *"What is the fee and hostel cost for B.Tech?"*), verified items must be answered, and unverified items explicitly flagged for confirmation. |
| **Rule 6** | **Admissions Direction** | When facts are unverified, directs the student to the Admission Cell without guessing. |
| **Rule 7** | **No Callback Promises** | The model is strictly forbidden from stating or implying that *"our team will call you back"*. |

---

## 4. Enquiry Form Trigger Policy (`app/llm/handler.py:L67-95`)

When an LLM response is returned, `GroqLLMHandler.detect_form_trigger()` scans the generated text using compiled regular expressions for triggers such as:
- `\b(?:enquiry|admission|application)\s+form\b`
- `\bsubmit\s+(?:an?\s+|your\s+)?(?:enquiry|query)\b`
- `\bcontact\s+(?:the\s+)?admissions\s+cell\b`
- `\b(?:cannot|unable\s+to)\s+(?:verify|confirm)\b`
- `\bunverified\b` / `\bnot\s+listed\b`

**System Action:**
If any pattern matches, the handler sets:
```python
show_enquiry_form = True
form_policy = "prompt"
```
The FastAPI backend passes this flag to `static/js/chatbot.js`, which automatically renders the interactive **Student Enquiry Modal** directly below the response card.

---

## 5. Network Resilience, Timeouts & Error Handling

To prevent the chatbot from freezing if the external Groq API is slow or unavailable:
1. **Configurable Timeout:** Fixed at `15.0` seconds (`settings.groq_timeout_seconds`).
2. **Retry Mechanism:** Up to `2` retries with exponential backoff on transient network faults (`groq.APIConnectionError`, `groq.RateLimitError`).
3. **Safe Local Fallback:** If the API fails completely, the system catches the exception and returns a structured fallback response:
   - `text`: *"I am currently unable to verify this detail through official university records. Please submit an enquiry to the DSU Admission Cell below."*
   - `show_enquiry_form = True`
   - `is_fallback = True`
   - Application status code: `HTTP 200 OK` (User experience does not crash with a 500 server error).
