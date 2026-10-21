# PART 5: END-TO-END EXECUTION TRACES
## Verifiable Execution Walkthroughs Across Four Real Scenarios

---

### Scenario 1: A Question Answered Locally via Local NLP
**Student Query:** `"What is the annual fee for the BCA programme at DSU?"`

```mermaid
sequenceDiagram
    autonumber
    actor Student
    participant UI as static/js/chatbot.js
    participant API as app/main.py (/chat)
    participant NLP as app/nlp/pipeline.py
    participant Prep as app/nlp/preprocessor.py
    participant FAQ as app/nlp/faq_retriever.py
    participant DB as PostgreSQL (Log)

    Student->>UI: Types "What is the annual fee for the BCA programme at DSU?"
    UI->>API: POST /chat {"message": "...", "session_id": "sess_101"}
    API->>NLP: pipeline.process_query(message, session_id)
    NLP->>Prep: normalize("What is the annual fee for the BCA programme at DSU?")
    Prep-->>NLP: "what is the annual fee for the bca program at dsu"
    NLP->>FAQ: find_best_match(normalized_text)
    Note over FAQ: TF-IDF Cosine Similarity = 0.94 (>= 0.65 threshold)
    FAQ-->>NLP: Matches FAQ: "What is the annual fee for the BCA programme at DSU?"
    NLP-->>API: ChatbotNLPResponse(text="For 2026-27, the yearly fee for BCA is INR 2,07,000...", intent="sca_fees", confidence=0.94, handler="local")
    API->>DB: INSERT into chat_interaction_logs (intent="sca_fees", conf=0.94)
    API-->>UI: HTTP 200 OK {response: "...", show_enquiry_form: false, sources: [...]}
    UI-->>Student: Renders fee breakdown card with official verified link
```

- **Runtime Decision:** The query directly matches a pre-verified entry in `faq_info.json`.
- **Groq LLM Invocation:** **NO.** The system bypasses external API calls entirely, answering with zero cloud latency (<15ms) and zero hallucination risk.

---

### Scenario 2: Different Wording for the Same Intent
**Student Query:** `"How much should I pay for Data Science?"`

1. **Preprocessing (`app/nlp/preprocessor.py`):**
   - `"How much should I pay for Data Science?"` $\rightarrow$ `"how much should i pay for data science"`
2. **Entity Extraction (`app/nlp/entity_extractor.py`):**
   - Matches program catalog: `course_id = "bca_ds"` (BCA in Data Science).
3. **Intent Classification (`app/nlp/classifier.py`):**
   - Pattern vectorizes via `FeatureUnion` (word n-grams + character n-grams).
   - Logistic Regression predicts: `tag = "sca_fees"` with probability `0.88` ($> 0.60$).
4. **Answer Construction (`app/nlp/pipeline.py`):**
   - Matches the extracted entity `bca_ds` with fee schedules in `college_info.json`:
   - *"The annual fee for BCA (Data Science) is INR 2,07,000 per year."*
5. **Why it works despite alternate phrasing:**
   - The TF-IDF model associates phrases like `"how much"`, `"pay"`, and `"fee"` with the `sca_fees` training class.
   - Character n-grams (`char_wb`) ensure robust token matching even if spelling deviates slightly.

---

### Scenario 3: Question Requiring LLM Fallback
**Student Query:** `"What degree programmes does the School of Engineering offer?"`

```mermaid
sequenceDiagram
    autonumber
    actor Student
    participant API as app/main.py
    participant NLP as app/nlp/pipeline.py
    participant Clf as app/nlp/classifier.py
    participant Ret as app/retrieval/dsu_retriever.py
    participant LLM as app/llm/handler.py (Groq)

    Student->>API: "What degree programmes does the School of Engineering offer?"
    API->>NLP: process_query()
    NLP->>Clf: predict()
    Clf-->>NLP: tag="other_department_courses", handler="llm", conf=0.82
    Note over NLP: Intent handler requires LLM fallback
    NLP->>Ret: retrieve_evidence(query, entity="School of Engineering")
    Ret-->>NLP: Grounding context (B.Tech in CSE, AI&ML, ECE, Mechanical, Aerospace...)
    NLP->>LLM: generate_dynamic_response(query, context)
    Note over LLM: Groq evaluates prompt with strict SYSTEM_PROMPT guardrails
    LLM-->>NLP: LLMResponse(text="**Answer:** The School of Engineering offers...", sources=["https://www.dsu.edu.in/..."])
    NLP-->>API: ChatbotNLPResponse(used_llm=True, show_enquiry_form=False)
    API-->>Student: Displays structured answer with official source citations
```

- **Runtime Decision:** Engineering courses are outside the primary local School of Computer Applications dataset. The classifier detects `other_department_courses` with `handler="llm"`.
- **Safety Control:** The Groq handler receives verified departmental catalog text via `DSURetriever`. It formats the answer into the mandatory 4-part structure (**Answer**, **Key details**, **Official sources**, **Next step**) and appends clickable official links.

---

### Scenario 4: Unsupported Detail Followed by Enquiry Lead Submission
**Student Query:** `"What is the management quota fee for M.Tech Robotics?"`

```mermaid
sequenceDiagram
    autonumber
    actor Student
    actor Staff as Admission Cell Staff
    participant UI as static/js/chatbot.js
    participant API as app/main.py
    participant NLP as app/nlp/pipeline.py
    participant LLM as app/llm/handler.py
    participant DB as PostgreSQL
    participant Dash as static/js/admin-dashboard.js

    Student->>UI: "What is the management quota fee for M.Tech Robotics?"
    UI->>API: POST /chat
    API->>NLP: process_query()
    NLP->>LLM: handler == "llm_fee_lookup"
    Note over LLM: Management quota for M.Tech Robotics is NOT published in public fee table
    LLM-->>NLP: "Specific management quota fees require confirmation. Please submit an enquiry..."
    Note over LLM: detect_form_trigger() detects "submit an enquiry"
    NLP-->>API: show_enquiry_form = True, form_policy = "prompt"
    API-->>UI: HTTP 200 OK {show_enquiry_form: true}
    UI-->>Student: Renders message + displays Student Enquiry Form Modal
    Student->>UI: Fills Form (Name: "Sneha Reddy", Email: "sneha@gmail.com", Phone: "9876543210")
    UI->>API: POST /enquiries {student_name: "Sneha Reddy", course_name: "M.Tech Robotics", ...}
    API->>DB: INSERT into enquiries (status='processing')
    API-->>UI: HTTP 201 Created {"enquiry_id": 42, "status": "processing"}
    UI-->>Student: "Your enquiry #42 has been submitted successfully!"
    
    Note over Staff,Dash: Staff checks portal on Port 5501
    Staff->>Dash: Logs in & views dashboard
    Dash->>API: GET /api/admin/enquiries
    API->>DB: SELECT * FROM enquiries WHERE status='processing'
    DB-->>API: Returns list including Enquiry #42
    API-->>Dash: Renders Enquiry #42 in table
    Staff->>Dash: Clicks "Manage" -> Updates status to 'called' -> Adds comment "Quoted fee structure"
    Dash->>API: PATCH /api/admin/enquiries/42/status {"status": "called"}
    API->>DB: UPDATE enquiries SET status='called' & INSERT into enquiry_status_history
    DB-->>API: Success
    API-->>Dash: Updated timeline rendered
```

- **Result:** Complete end-to-end integration: unanswered student question automatically converts into a persistent lead in PostgreSQL, which is immediately actionable by university admission personnel on the staff dashboard.
