# PART 4: FILE-BY-FILE IMPLEMENTATION MAPPING
## Comprehensive Structural and Functional Breakdown

---

### 1. `app/main.py`
- **Location:** `app/main.py`
- **Purpose:** Central application entry point, lifecycle manager, CORS middleware configurator, and router coordinator.
- **Dependencies:** `fastapi`, `uvicorn`, `sqlalchemy.orm`, `app.config`, `app.db`, `app.nlp.pipeline`, `app.api.admin`.
- **Key Functions / Objects:**
  - `lifespan(app: FastAPI)`: Pre-trains/loads the NLP singleton into memory and runs DB table checks and default admin seeding.
  - `get_pipeline() -> NLPPipeline`: Thread-safe dependency returning the singleton pipeline instance.
  - `app`: The instantiated FastAPI object.
  - `POST /chat`: Receives incoming `ChatRequest` and delegates to `pipeline.process_query()`.
  - `POST /enquiries`: Validates and persists student lead submissions to PostgreSQL.
- **Request Flow Role:** Top-level controller orchestrating every HTTP request.

---

### 2. `app/config.py`
- **Location:** `app/config.py`
- **Purpose:** Centralized, type-validated settings loaded from `.env` via Pydantic v2 `BaseSettings`.
- **Key Variables:**
  - `database_url`: PostgreSQL connection string (`postgresql+psycopg://...`).
  - `groq_api_key`, `groq_model`: Cloud LLM parameters.
  - `intent_confidence_threshold`: Minimum probability score required (`0.60`).
  - `faq_confidence_threshold`: Minimum cosine similarity score required (`0.65`).
  - `context_ttl_turns`: Conversation memory lifespan (`6` turns).
- **Security Role:** Enforces validation on startup and prevents hardcoded secrets in the codebase.

---

### 3. `app/nlp/preprocessor.py`
- **Location:** `app/nlp/preprocessor.py`
- **Purpose:** Cleans, normalizes, and tokenizes raw conversational queries into standardized strings.
- **Key Functions / Dictionaries:**
  - `normalize(text: str) -> str`: Executes NFKD Unicode normalization, lowercasing, contraction expansion, degree regex mapping, punctuation stripping, and whitespace collapsing.
  - `tokenize(text: str) -> List[str]`: Generates a list of word tokens via `.split()`.
  - `degree_patterns`: 18 compiled regular expressions mapping academic degree variations (e.g., `B.Tech.` -> `btech`).
  - `domain_synonyms`: Regex substitutions for frequent student typos (`scince`, `hostle`, `eligiblity`).

---

### 4. `app/nlp/classifier.py`
- **Location:** `app/nlp/classifier.py`
- **Purpose:** Machine learning intent classification engine utilizing Scikit-learn pipelines.
- **Key Functions / Classes:**
  - `IntentClassifier`: Main classifier class.
  - `_build_pipeline() -> Pipeline`: Assembles a `FeatureUnion` combining word n-grams `(1, 2)` and character n-grams `(2, 4)` feeding into `LogisticRegression(C=5.0, class_weight='balanced')`.
  - `fit(intents_data)`: Fits the pipeline dynamically on all patterns defined in `intents.json`.
  - `predict(text: str) -> IntentPrediction`: Evaluates class probabilities, checks against the `0.60` threshold, and detects query ambiguity (`margin <= 0.08`).

---

### 5. `app/nlp/faq_retriever.py`
- **Location:** `app/nlp/faq_retriever.py`
- **Purpose:** Semantic FAQ matching using TF-IDF term vectors and cosine similarity against `faq_info.json`.
- **Key Functions / Classes:**
  - `FAQRetriever`: Class storing in-memory matrix of vectorized FAQ questions.
  - `find_best_match(query: str) -> Optional[FAQMatchResult]`: Calculates cosine distance between query vector and FAQ matrix. Returns verified answers when similarity $\ge 0.65$.

---

### 6. `app/nlp/entity_extractor.py`
- **Location:** `app/nlp/entity_extractor.py`
- **Purpose:** Extracts degrees, branches, departments, and quota types from queries using longest-match substring catalog matching.
- **Key Functions / Classes:**
  - `EntityExtractor.extract(text: str) -> ExtractedEntities`: Returns structured metadata (`degree`, `branch`, `school`, `quota`).
  - Eliminates probabilistic uncertainty by matching directly against the official `college_info.json` academic catalog.

---

### 7. `app/nlp/context_tracker.py`
- **Location:** `app/nlp/context_tracker.py`
- **Purpose:** Maintains in-memory session states to enable coherent multi-turn conversations.
- **Key Functions / Classes:**
  - `ContextTracker`: Manages session contexts dictionary.
  - `ConversationSlotContext`: Holds active entity slots (`last_course_id`, `last_degree`, `last_school`).
  - `step_turn(session_id)`: Increments turn counter; clears slots when turn count exceeds `6` without updates.

---

### 8. `app/nlp/pipeline.py`
- **Location:** `app/nlp/pipeline.py`
- **Purpose:** Master coordinator executing the complete natural language workflow for every question.
- **Key Functions / Classes:**
  - `NLPPipeline.process_query(message, session_id) -> ChatbotNLPResponse`: Orchestrates preprocessing, entity extraction, context tracking, FAQ matching, intent classification, dynamic retrieval, and LLM fallback escalation.

---

### 9. `app/llm/handler.py`
- **Location:** `app/llm/handler.py`
- **Purpose:** Grounded fallback handler interfacing with the Groq Cloud API.
- **Key Functions / Classes:**
  - `GroqLLMHandler`: Async API wrapper.
  - `SYSTEM_PROMPT`: Injects 7 non-negotiable guardrails prohibiting hallucination, callback promises, and fee substitution.
  - `detect_form_trigger(text: str) -> bool`: Regex engine that checks whether the LLM indicated missing information or recommended an enquiry form.

---

### 10. `app/retrieval/dsu_retriever.py`
- **Location:** `app/retrieval/dsu_retriever.py`
- **Purpose:** Structured evidence retrieval assembling verified institutional facts from `college_info.json` to feed into LLM prompt contexts.
- **Key Classes:** `DSURetriever`, `RetrievedEvidence`.

---

### 11. `app/db/models.py`
- **Location:** `app/db/models.py`
- **Purpose:** SQLAlchemy declarative ORM models defining the database schema in PostgreSQL.
- **Key Models:**
  - `StaffUser`: Staff accounts with salted password hashes (`hashed_password`, `salt`, `role`).
  - `Enquiry`: Student leads (`student_name`, `email`, `phone_number`, `course_name`, `status`, `assigned_staff_id`).
  - `EnquiryComment`: Internal staff notes linked to an enquiry.
  - `EnquiryStatusHistory`: Audit trail of all lifecycle status changes (`processing` -> `on_hold` -> `called` -> `completed`).
  - `EnquiryFollowUp`: Scheduled tasks (callbacks, campus tours, verification).
  - `ChatInteractionLog`: Telemetry logs storing query intent and confidence without student personal data.

---

### 12. `app/api/admin.py`
- **Location:** `app/api/admin.py`
- **Purpose:** Protected REST API endpoints supporting the Admission Cell CRM portal.
- **Key Endpoints:**
  - `POST /api/admin/login`: Verifies credentials and issues HTTP-only JWT cookies.
  - `GET /api/admin/enquiries`: Paginated query returning student leads with filters for search, course, and status.
  - `PATCH /api/admin/enquiries/{id}/status`: Updates lead status and writes an audit log.
  - `POST /api/admin/enquiries/{id}/comments`: Appends internal notes.
  - `POST /api/admin/enquiries/{id}/follow-ups`: Schedules phone calls or campus visits.

---

### 13. `static/js/chatbot.js`
- **Location:** `static/js/chatbot.js`
- **Purpose:** Client-side controller for the student website widget.
- **Key Functions:**
  - Manages session IDs, triggers typing animations, dispatches `POST /chat` fetch requests, dynamically renders message cards, format tables, action chips, and presents the lead enquiry modal.

---

### 14. `static/js/admin-dashboard.js`
- **Location:** `static/js/admin-dashboard.js`
- **Purpose:** Frontend script for the Admission Cell staff portal.
- **Key Functions:**
  - Handles authentication tokens, populates analytical metrics, executes asynchronous searches across enquiries, and manages modal dialogues for updating statuses, writing notes, and scheduling follow-up calls.
