"""Dynamic Official DSU Knowledge Retriever.

Performs live search and content extraction from official Dayananda Sagar University
webpages (https://www.dsu.edu.in/) with an authoritative fallback catalog for offline
and timeout resilience.

Strictly preserves official URLs and extracts factual text evidence for LLM grounding.
"""

import html
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class RetrievedEvidence:
    """A unit of verified evidence retrieved from official DSU sources."""
    title: str
    url: str
    snippet: str
    content: str
    relevance_score: float = 1.0
    source_type: str = "official_web"  # "official_web" or "official_catalog"


class DSURetriever:
    """Searches official DSU web pages and extracts grounded evidence for student queries."""

    BASE_URL: str = "https://www.dsu.edu.in"
    DEFAULT_TIMEOUT: float = 2.5
    CACHE_TTL_SECONDS: float = 3600.0  # 1 hour cache

    # Curated authoritative official catalog providing instant offline resilience and rich factual backing
    OFFICIAL_CATALOG: List[Dict[str, Any]] = [
        {
            "title": "School of Computer Applications - Dean's Message & Leadership | DSU",
            "url": "https://www.dsu.edu.in/computer-applications/dean-msg-sca",
            "keywords": ["dean", "senthil", "leadership", "head", "sca", "computer applications", "director", "professor", "designation", "faculty"],
            "summary": (
                "Dr. S. Senthil is the Professor and Dean of the School of Computer Applications (SCA) at Dayananda Sagar University. "
                "The School of Computer Applications, located at the Kudlu Gate City Innovation Campus on Hosur Road, Bengaluru, offers premier programs including BCA, B.Sc. Data Science, and MCA. "
                "Under Dr. S. Senthil's leadership as Dean and Chairman, the school emphasizes cutting-edge curricula in artificial intelligence, data science, cloud computing, and software development, "
                "partnering with leading industry organizations including NVIDIA, Autodesk, and IBM."
            ),
        },
        {
            "title": "B.Sc. Data Science - Program Overview, Eligibility & Placements | DSU",
            "url": "https://www.dsu.edu.in/computer-applications/bsc-data-science",
            "keywords": ["bsc data science", "b.sc data science", "data science", "bsc ds", "data science fee", "data science eligibility", "data science placement", "data science careers", "data science duration"],
            "summary": (
                "B.Sc. Data Science at the School of Computer Applications, DSU (Kudlu Gate Campus, Bengaluru). "
                "Duration: 3 Years (6 Semesters). "
                "Academic Year 2026-27 Tuition Fee: Total annual fee is INR 1,62,000 (payable in two instalments: Term 1: INR 87,000 and Term 2: INR 75,000). "
                "Eligibility: Pass in 10+2 / Pre-University (PUC) examination or equivalent with Mathematics / Computer Science / Statistics / equivalent subjects with at least 50% aggregate marks. "
                "Career Opportunities: Data Scientist, Machine Learning Associate, Data Analyst, Business Intelligence Engineer, Big Data Specialist, and Data Architect. "
                "Placement Statistics: Placement percentage is over 90%, with highest CTC reaching 30+ LPA and competitive average packages across tech and analytics recruiters (Autodesk, Bosch, IBM, Cisco, Microsoft, Infosys)."
            ),
        },
        {
            "title": "Admissions Documentation & Verification | DSU",
            "url": "https://www.dsu.edu.in/admission",
            "keywords": ["documents", "verification", "marks card", "tc", "transfer certificate", "migration", "admission documents", "12th mec", "mec stream", "mec", "prepare documents"],
            "summary": (
                "DSU Admission Verification & Mandatory Documents: "
                "For Undergraduate Admission (e.g., BCA, B.Sc. Data Science): "
                "1. 10th Standard / SSLC Marks Card (proof of date of birth). "
                "2. 12th Standard / 2nd PUC Marks Card (Candidates with 12th MEC - Mathematics, Economics, Commerce - or Science/Commerce streams with Mathematics/Computer Science with minimum 45% aggregate are eligible for BCA; having 60% comfortably meets the eligibility requirement). "
                "3. Transfer Certificate (TC) issued by the previous institution. "
                "4. Migration Certificate (for students from non-Karnataka / CBSE / ICSE / Other State Boards). "
                "5. Conduct / Study Certificate. "
                "6. Copy of Aadhaar Card / Identity Proof. "
                "7. Recent passport-size colour photographs (5-6 copies). "
                "8. Caste / Category / Income Certificate (if claiming reservations or fee concessions)."
            ),
        },
        {
            "title": "DSU Fee Structure Breakdown - Tuition, Hostel, Mess & Transport | DSU",
            "url": "https://www.dsu.edu.in/eligibility",
            "keywords": ["difference between tuition", "hostel fee mess fee", "fee components", "mess charges", "transport fee", "fee breakdown", "explain the difference"],
            "summary": (
                "Dayananda Sagar University Fee Component Breakdown: "
                "1. Tuition Fee: Covers academic instruction, classroom teaching, laboratory access, faculty guidance, and core university academic infrastructure for the chosen program. Billed per academic year (divided into Term 1 and Term 2). "
                "2. Hostel (Accommodation) Fee: Covers residential room occupancy, maintenance, water, electricity, Wi-Fi, and residential campus amenities (separate blocks for boys and girls primarily at Harohalli Main Campus). "
                "3. Mess Fee: Covers nutritious multi-cuisine meal plans (breakfast, lunch, evening snacks, and dinner) prepared in hygienic centralized dining facilities. Billed as part of the overall hostel boarding structure. "
                "4. Transport (Bus) Fee: Optional annual fee for daily pick-and-drop bus commute covering major transit corridors across Bengaluru to campus. Charged based on the designated pickup route/distance. "
                "Note: Hostel, mess, and transport fees are auxiliary living and commute charges separate from program academic tuition."
            ),
        },
        {
            "title": "Course Eligibility & Fee Structure | Dayananda Sagar University",
            "url": "https://www.dsu.edu.in/eligibility",
            "keywords": ["fee", "fees", "eligibility", "tuition", "cost", "annual", "bca", "mca", "btech", "mba", "criteria", "refund"],
            "summary": (
                "Official Course Eligibility and Fee Schedules for Dayananda Sagar University. "
                "For School of Computer Applications: BCA annual tuition fee is INR 2,07,000 (Term 1: INR 1,09,500; Term 2: INR 97,500); MCA annual tuition fee is INR 2,07,000; B.Sc. Data Science annual tuition fee is INR 1,62,000 (Term 1: INR 87,000; Term 2: INR 75,000). "
                "Eligibility for BCA: 10+2 / Pre-University equivalent with mathematics, computer science, or equivalent with at least 50% marks (45% for general minimum). "
                "Eligibility for MCA: Recognized Bachelor's degree (BCA, B.Sc, B.Com, B.A) with mathematics at 10+2 or degree level with at least 50% aggregate. "
                "For Engineering (B.Tech), Commerce (BBA, B.Com), Law, Health Sciences (Pharmacy, Nursing, Physiotherapy), and other schools, "
                "fees vary by entry quota (Government CET, COMED-K, PGCET, or Management quota) and chosen specialisation. "
                "Fee payment schedules are per academic year. Fee refund policies strictly adhere to statutory UGC notifications."
            ),
        },
        {
            "title": "Admissions 2026 - Application Procedures & Entrance Exams | DSU",
            "url": "https://www.dsu.edu.in/admission",
            "keywords": ["admission", "admissions", "apply", "application", "entrance", "exam", "dsat", "cet", "comedk", "pgcet", "dates", "procedure"],
            "summary": (
                "DSU Admissions for 2026 Academic Year. Students can apply online through the official portal (https://admissions.dsu.edu.in). "
                "Admission pathways include: "
                "1. Dayananda Sagar Admission Test (DSAT) for national merit applicants. "
                "2. Karnataka CET (KCET) - Code E240 for B.Tech programs. "
                "3. COMED-K - Code E182 for Engineering programs. "
                "4. Karnataka PGCET - Code T970 for M.Tech, Code B365MB for MBA, Code C520MC for MCA. "
                "Admission office helplines: 080 46461800 / +91 6366885507. Email: admissions@dsu.edu.in."
            ),
        },
        {
            "title": "Hostel Accommodation, Fees & Facilities | DSU Main Campus",
            "url": "https://www.dsu.edu.in/9-explore/1575-hostel-fee",
            "keywords": ["hostel", "hostels", "accommodation", "room", "mess", "food", "living", "stay", "harohalli", "kudlu"],
            "summary": (
                "Official Hostel Facilities and Fee Structure for Dayananda Sagar University. "
                "DSU operates student hostel accommodation primarily at the Main Campus in Harohalli (Kanakapura Road). "
                "Hostel facilities include separate blocks for male and female students, nutritious multi-cuisine mess (food), "
                "24/7 security, high-speed Wi-Fi, laundry services, study halls, gym, and healthcare center. "
                "Total fee payable covers accommodation, messing, and laundry per academic year. "
                "Hostel accommodation at Kudlu Gate City Innovation Campus is unconfirmed in primary circulars; "
                "students taking classes at Kudlu Gate should contact the hostel administration desk for verified local lodging."
            ),
        },
        {
            "title": "Placements, Recruiters & Career Development Centre | DSU",
            "url": "https://www.dsu.edu.in/placements/overview-placements",
            "keywords": ["placement", "placements", "recruiters", "salary", "package", "job", "career", "hiring", "internship", "companies", "highest"],
            "summary": (
                "Centre for Executive Development and Career Planning (CEDC) at DSU coordinates comprehensive career placements and corporate internships. "
                "Placement Track Record: Over 90% placement record with multinational technology, engineering, and corporate finance companies. "
                "Top Recruiters: Autodesk, Bosch Rexroth, Microsoft, Amazon, Cisco, IBM, VMware, Intel, NVIDIA, Infosys, Wipro, Accenture, Cognizant, Mercedes-Benz. "
                "The highest CTC offers reach 30+ LPA with competitive average packages across computer science, engineering, and management programs. "
                "Pre-placement training includes mock interviews, coding bootcamps, and industry certification academies on campus."
            ),
        },
        {
            "title": "Scholarships & Financial Assistance Schemes | DSU Admissions",
            "url": "https://www.dsu.edu.in/106-admission/306-scholarship",
            "keywords": ["scholarship", "scholarships", "financial", "aid", "waiver", "concession", "merit", "sports", "defense"],
            "summary": (
                "Dayananda Sagar University offers competitive scholarships and financial assistance programs: "
                "1. DSAT Merit Scholarships: Awarded to top-ranking students appearing in the national DSAT entrance exam. "
                "2. Defence & Paramilitary Personnel Ward Concessions: Fee waivers for children of armed forces personnel. "
                "3. Sports Achievements Scholarships: Fee concessions for national and state-level sports achievers. "
                "4. Merit-cum-means assistance: Facilitation of state and central government post-matric scholarship schemes for reserved category students. "
                "Scholarship continuation requires maintaining prescribed SGPA/CGPA and clean disciplinary standing."
            ),
        },
        {
            "title": "Academic Calendar, Examination Regulations & Attendance | DSU",
            "url": "https://www.dsu.edu.in/examination",
            "keywords": ["exam", "exams", "examination", "calendar", "attendance", "rules", "regulations", "grading", "semester", "cgpa", "sgpa"],
            "summary": (
                "Office of the Controller of Examinations (COE) at DSU governs semester assessments and evaluation. "
                "Evaluation Pattern: Choice-Based Credit System (CBCS) comprising Continuous Internal Evaluation (CIE) and Semester End Examinations (SEE). "
                "Attendance Regulation: A minimum of 75% attendance in each course is strictly mandatory to be eligible to appear for the semester end examinations. "
                "Odd semesters typically run from August to December; even semesters run from January to May. "
                "Examinations follow university standard grading on a 10-point scale."
            ),
        },
        {
            "title": "NVIDIA Academic AI Factory & Research Centers | DSU",
            "url": "https://www.dsu.edu.in/nvidia-ai-factory",
            "keywords": ["nvidia", "ai", "artificial intelligence", "research", "factory", "lab", "supercomputing", "innovation", "aic"],
            "summary": (
                "DSU has partnered with NVIDIA to establish India's First Academic AI Factory, equipping the university with sovereign AI supercomputing clusters. "
                "Research Ecosystem: AIC-DSU Innovation Foundation features over 4 lakh sq. ft. of space supporting 200+ technology enterprises and student startups. "
                "Advanced industry training academies on campus include Autodesk, Bosch Etas, IBM, VMware, and Intel. "
                "Research centers specialize in generative AI, data mining, robotics, healthcare computing, composite materials, and thin films."
            ),
        },
        {
            "title": "Transportation, Campus Fleet & Route Network | DSU",
            "url": "https://www.dsu.edu.in/campus-life",
            "keywords": ["transport", "bus", "transportation", "buses", "route", "routes", "pickup", "commute", "travel"],
            "summary": (
                "University Transportation Services: DSU operates a fleet of buses connecting the Harohalli Main Campus "
                "with major transit hubs across Bengaluru, including Banashankari, Majestic, Electronic City, Jayanagar, BTM Layout, and Kanakapura. "
                "Bus routes are scheduled to align with university class timings. "
                "Transport fees are billed annually and depend on the designated pickup stage/zone. "
                "Bus passes must be obtained at the commencement of each academic year from the transport administrative office."
            ),
        },
        {
            "title": "Official University Campuses & Administrative Locations | DSU",
            "url": "https://www.dsu.edu.in/about-us/dsu",
            "keywords": ["campus", "campuses", "location", "locations", "address", "main", "kudlu", "harohalli", "kumaraswamy", "contact"],
            "summary": (
                "Dayananda Sagar University functions across three key campuses in Bengaluru, Karnataka: "
                "1. Main Campus: Devarakaggalahalli, Harohalli, Kanakapura Road, Bengaluru South – 562 112. (Houses School of Engineering, School of Basic & Applied Sciences, CDSIMER, primary sports complexes, and student residential hostels). "
                "2. City Innovation Campus: Kudlu Gate, Hosur Road, Bengaluru – 560 114. (Houses School of Computer Applications, School of Commerce & Management Studies, and AIC-DSU Innovation Centre). "
                "3. City Admissions Office: Gate 2, 6th Floor, Dental Block, Kumaraswamy Layout, Bengaluru – 560 111. "
                "Central Helpline: 080 46461800 / +91 6366885507."
            ),
        },
        {
            "title": "School of Law - Programs & Moot Court Training | DSU",
            "url": "https://www.dsu.edu.in/academics/schools/law",
            "keywords": ["law", "llb", "bba llb", "ba llb", "llm", "advocate", "bar", "moot", "court", "legal"],
            "summary": (
                "School of Law at Dayananda Sagar University offers B.A. LL.B (Hons), B.B.A. LL.B (Hons), 3-Year LL.B, and LL.M programs. "
                "Approved by the Bar Council of India (BCI). Facilities include state-of-the-art moot court halls, legal aid clinics, "
                "and judicial mentorship programs. Contact admissions for School of Law: 9606484800."
            ),
        },
    ]

    def __init__(self, timeout: float = DEFAULT_TIMEOUT):
        self.timeout = timeout
        self._cache: Dict[str, Tuple[float, List[RetrievedEvidence]]] = {}
        self._page_cache: Dict[str, Tuple[float, str]] = {}

    def _extract_search_keywords(self, query: str) -> List[str]:
        """Extracts prioritized topic keywords from student query for site search."""
        stop_words = {
            "what", "is", "the", "are", "of", "for", "at", "dsu", "in", "to", "how",
            "can", "i", "get", "tell", "me", "about", "please", "do", "you", "have",
            "does", "a", "an", "and", "or", "on", "any", "which", "there", "give", "details",
            "information", "want", "know", "my", "your", "by", "from", "with", "as"
        }
        words = [w for w in re.findall(r"\b[a-zA-Z0-9_\-\.]{2,}\b", query.lower()) if w not in stop_words]
        return words

    def _clean_html_text(self, raw_html: str) -> str:
        """Strips HTML tags, scripts, and chrome to extract legible text."""
        if not raw_html:
            return ""
        # Remove script and style tags
        cleaned = re.sub(r"<(script|style|nav|header|footer)[^>]*>.*?</\1>", " ", raw_html, flags=re.DOTALL | re.IGNORECASE)
        # Strip all HTML tags
        cleaned = re.sub(r"<[^>]+>", " ", cleaned)
        # Decode HTML entities
        cleaned = html.unescape(cleaned)
        # Normalize whitespace
        cleaned = " ".join(cleaned.split())
        return cleaned

    def fetch_page_content(self, url: str) -> Optional[str]:
        """Fetches live page content from official DSU domain with in-memory caching."""
        now = time.time()
        if url in self._page_cache:
            ts, cached_content = self._page_cache[url]
            if now - ts < self.CACHE_TTL_SECONDS:
                return cached_content

        try:
            import httpx
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) DSU-Academic-Enquiry/1.0"}
            resp = httpx.get(url, timeout=self.timeout, follow_redirects=True, headers=headers)
            if resp.status_code == 200:
                clean_text = self._clean_html_text(resp.text)
                self._page_cache[url] = (now, clean_text)
                return clean_text
        except Exception as e:
            logger.warning("Live page fetch failed for %s: %s", url, type(e).__name__)

        return None

    def search_live_dsu(self, keyword: str, max_items: int = 3) -> List[RetrievedEvidence]:
        """Queries DSU official Joomla Finder search endpoint for the given keyword."""
        search_url = f"{self.BASE_URL}/component/finder/search?q={keyword}"
        evidences: List[RetrievedEvidence] = []

        try:
            import httpx
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) DSU-Academic-Enquiry/1.0"}
            resp = httpx.get(search_url, timeout=self.timeout, follow_redirects=True, headers=headers)
            if resp.status_code != 200:
                return []

            # Match search result items in DSU Joomla template
            item_pattern = re.compile(
                r'<h4[^>]*class=["\'][^"\']*result-title[^"\']*["\'][^>]*>\s*<a[^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>'
                r'.*?(?:<p[^>]*class=["\'][^"\']*result-text[^"\']*["\'][^>]*>(.*?)</p>)?',
                re.DOTALL | re.IGNORECASE
            )
            matches = item_pattern.findall(resp.text)

            for href, raw_title, raw_snippet in matches[:max_items]:
                title = self._clean_html_text(raw_title).strip()
                snippet = self._clean_html_text(raw_snippet or "").strip()
                full_url = href if href.startswith("http") else f"{self.BASE_URL}{href}"

                # If snippet has substantial text (>= 80 chars), use it directly;
                # otherwise fetch full page content for richer context
                if len(snippet) >= 80:
                    body_content = snippet
                else:
                    page_text = self.fetch_page_content(full_url)
                    body_content = page_text[:1200] if page_text else snippet

                evidences.append(RetrievedEvidence(
                    title=title,
                    url=full_url,
                    snippet=snippet,
                    content=body_content,
                    relevance_score=1.0,
                    source_type="official_web"
                ))

        except Exception as e:
            logger.warning("Live DSU search encountered %s: %s", type(e).__name__, e)

        return evidences

    def search_catalog_fallback(self, query: str, max_items: int = 3) -> List[RetrievedEvidence]:
        """Searches the verified official DSU catalog when live search is offline or yields 0 results."""
        query_words = set(self._extract_search_keywords(query))
        scored_entries: List[Tuple[float, Dict[str, Any]]] = []

        for entry in self.OFFICIAL_CATALOG:
            keywords = entry.get("keywords", [])
            overlap = query_words.intersection(set(keywords))
            score = len(overlap)
            # Bonus for title or summary matches
            for word in query_words:
                if word in entry["title"].lower():
                    score += 1.5
                if word in entry["summary"].lower():
                    score += 0.5
            if score > 0:
                scored_entries.append((score, entry))

        scored_entries.sort(key=lambda x: x[0], reverse=True)

        results: List[RetrievedEvidence] = []
        for score, entry in scored_entries[:max_items]:
            results.append(RetrievedEvidence(
                title=entry["title"],
                url=entry["url"],
                snippet=entry["summary"][:200] + "...",
                content=entry["summary"],
                relevance_score=score,
                source_type="official_catalog"
            ))

        return results

    def _is_relevant_evidence(self, query: str, evidence: RetrievedEvidence) -> bool:
        """Validates that retrieved evidence is genuinely relevant to the query topic."""
        q_lower = query.lower()
        title_lower = (evidence.title or "").lower()
        content_lower = (evidence.content or "").lower()
        url_lower = (evidence.url or "").lower()
        text_corpus = f"{title_lower} {content_lower} {url_lower}"

        # Disallow personal faculty bios, staff directory, or random event pages when query is about facilities, fees, or placements
        disallow_faculty = any(term in q_lower for term in [
            "fee", "fees", "cost", "tuition", "hostel", "mess", "accommodation",
            "placement", "package", "recruiter", "scholarship", "admission", "transport"
        ])
        if disallow_faculty and any(p in url_lower for p in ["/mohd-", "/faculty/", "/prof-", "/dr-", "/staff-", "events-computer-applications"]):
            if not any(f in q_lower for f in ["faculty", "dean", "professor", "teacher", "lecturer"]):
                return False

        # Topic 1: Hostel / Accommodation / Mess
        if any(term in q_lower for term in ["hostel", "accommodation", "mess", "room", "stay", "living"]):
            if not any(term in text_corpus for term in ["hostel", "accommodation", "mess", "living"]):
                return False

        # Topic 2: Placements / Package / Recruiters
        if any(term in q_lower for term in ["placement", "package", "salary", "recruiter", "ctc", "hiring", "highest", "average"]):
            if not any(term in text_corpus for term in ["placement", "recruiter", "salary", "package", "career", "company", "companies"]):
                return False

        # Topic 3: Fee / Tuition / Quota
        if any(term in q_lower for term in ["fee", "fees", "tuition", "quota", "cost", "management quota", "comedk", "cet"]):
            if not any(term in text_corpus for term in ["fee", "fees", "tuition", "eligibility", "admission", "quota"]):
                return False

        # Topic 4: Scholarships
        if any(term in q_lower for term in ["scholarship", "scholarships", "fee concession", "waiver"]):
            if not any(term in text_corpus for term in ["scholarship", "concession", "waiver", "financial aid", "merit"]):
                return False

        # Topic 5: Transport
        if any(term in q_lower for term in ["transport", "bus", "buses", "route"]):
            if not any(term in text_corpus for term in ["transport", "bus", "route", "fleet", "commute"]):
                return False

        return True

    def search(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        max_results: int = 3
    ) -> List[RetrievedEvidence]:
        """Primary retrieval method searching live official DSU pages with catalog fallback and relevance filtering."""
        now = time.time()
        cache_key = query.strip().lower()

        # Check in-memory cache
        if cache_key in self._cache:
            ts, cached_res = self._cache[cache_key]
            if now - ts < self.CACHE_TTL_SECONDS:
                return cached_res

        keywords = self._extract_search_keywords(query)
        if context and context.get("course_id"):
            keywords.insert(0, str(context["course_id"]).replace("sca_", ""))

        live_results: List[RetrievedEvidence] = []

        # 1. Try Live DSU Web Search if keywords exist
        if keywords:
            multi_term = "+".join(keywords[:2])
            live_results = self.search_live_dsu(multi_term, max_items=max_results)

            if not live_results and len(keywords) > 0:
                live_results = self.search_live_dsu(keywords[0], max_items=max_results)

        # 2. Filter live search results by strict topic relevance
        valid_live = [ev for ev in live_results if self._is_relevant_evidence(query, ev)]

        # 3. Search verified authoritative official catalog
        catalog_results = self.search_catalog_fallback(query, max_items=max_results)
        valid_catalog = [ev for ev in catalog_results if self._is_relevant_evidence(query, ev)]

        # 4. Merge results: prioritizing verified official catalog for primary operational topics
        combined: List[RetrievedEvidence] = []
        seen_urls = set()

        for cat_ev in valid_catalog:
            if cat_ev.url not in seen_urls:
                combined.append(cat_ev)
                seen_urls.add(cat_ev.url)

        for live_ev in valid_live:
            if live_ev.url not in seen_urls:
                combined.append(live_ev)
                seen_urls.add(live_ev.url)

        final_results = combined[:max_results]
        if not final_results:
            final_results = (valid_catalog or catalog_results)[:max_results]

        # Cache results
        self._cache[cache_key] = (now, final_results)
        return final_results
