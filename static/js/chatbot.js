/**
 * DSU Enquiry Chatbot Client Controller
 * Connects the floating chat widget to existing FastAPI endpoints:
 * - POST /chat: Multi-turn NLP query processing, Groq fallback & enquiry policy
 * - POST /enquiries: Validation and storage of student enquiry submissions
 */

document.addEventListener("DOMContentLoaded", () => {
  initChatbotWidget();
});

function initChatbotWidget() {
  const launcherBtn = document.getElementById("dsu-chatbot-launcher");
  const chatPanel = document.getElementById("dsu-chatbot-panel");
  const closeBtn = document.getElementById("chat-close-btn");
  const minimizeBtn = document.getElementById("chat-minimize-btn");
  const chatInput = document.getElementById("chat-input");
  const sendBtn = document.getElementById("chat-send-btn");
  const messagesContainer = document.getElementById("chat-messages-container");
  const quickPromptBtns = document.querySelectorAll(".quick-prompt-btn");

  if (!launcherBtn || !chatPanel) return;

  const API_BASE = window.location.port === '8000' ? '' : 'http://127.0.0.1:8000';

  // Session ID tracking across multi-turn conversation
  let sessionId = sessionStorage.getItem("dsu_chat_session_id");
  if (!sessionId) {
    sessionId = "dsu_" + Math.random().toString(36).substring(2, 11) + "_" + Date.now();
    sessionStorage.setItem("dsu_chat_session_id", sessionId);
  }

  // 1. Toggle Chat Panel
  function toggleChat(open) {
    const isCurrentlyActive = chatPanel.classList.contains("active");
    const targetState = typeof open === "boolean" ? open : !isCurrentlyActive;

    if (targetState) {
      chatPanel.classList.add("active");
      launcherBtn.classList.add("open");
      if (chatInput) chatInput.focus();
    } else {
      chatPanel.classList.remove("active");
      launcherBtn.classList.remove("open");
    }
  }

  launcherBtn.addEventListener("click", () => toggleChat());
  if (closeBtn) closeBtn.addEventListener("click", () => toggleChat(false));
  if (minimizeBtn) minimizeBtn.addEventListener("click", () => toggleChat(false));

  // 2. Quick Starters Click Handlers
  quickPromptBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const promptText = btn.getAttribute("data-prompt") || btn.textContent.trim();
      sendMessage(promptText);
    });
  });

  // 3. Input Submission (Enter key and Send button)
  if (sendBtn && chatInput) {
    sendBtn.addEventListener("click", () => {
      const msg = chatInput.value.trim();
      if (msg) {
        sendMessage(msg);
        chatInput.value = "";
      }
    });

    chatInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        const msg = chatInput.value.trim();
        if (msg) {
          sendMessage(msg);
          chatInput.value = "";
        }
      }
    });
  }

  // 4. Message Dispatch to POST /chat
  async function sendMessage(userMessage) {
    if (!userMessage) return;

    // Append User Message to thread
    appendMessage("user", userMessage);
    scrollToBottom();

    // Show Typing Indicator
    const typingIndicator = createTypingIndicator();
    messagesContainer.appendChild(typingIndicator);
    scrollToBottom();

    try {
      const response = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: userMessage,
          session_id: sessionId,
        }),
      });

      // Remove typing indicator
      if (typingIndicator && typingIndicator.parentNode) {
        typingIndicator.parentNode.removeChild(typingIndicator);
      }

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        const errDetail = errData.detail || "Unable to reach the chatbot server. Please check your connection.";
        appendMessage("bot", `⚠️ ${errDetail}`);
        scrollToBottom();
        return;
      }

      const data = await response.json();

      // Render Bot Message Bubble with Markdown links & formatting
      appendMessage("bot", data.response_text, {
        source: data.source,
        quickReplies: data.quick_replies,
        showEnquiryForm: data.show_enquiry_form,
        formPrefill: data.form_prefill,
      });

      scrollToBottom();
    } catch (err) {
      if (typingIndicator && typingIndicator.parentNode) {
        typingIndicator.parentNode.removeChild(typingIndicator);
      }
      appendMessage(
        "bot",
        "Connection error: Unable to communicate with the enquiry backend. Please verify the service is running."
      );
      scrollToBottom();
    }
  }

  // 5. Message DOM Construction
  function appendMessage(sender, text, options = {}) {
    const row = document.createElement("div");
    row.className = `chat-msg-row ${sender}`;

    const bubble = document.createElement("div");
    bubble.className = "chat-bubble";

    // Format Markdown links: [Title](URL) -> <a href="URL" target="_blank">Title</a>
    const formattedHtml = formatMarkdown(text);
    bubble.innerHTML = formattedHtml;

    // Source Citation: Only display if options.source is an HTTP URL AND the text does not already contain a link to it
    if (options.source && options.source.startsWith("http") && !formattedHtml.includes(options.source)) {
      const sourceLink = document.createElement("a");
      sourceLink.className = "source-citation";
      sourceLink.href = options.source;
      sourceLink.target = "_blank";
      sourceLink.rel = "noopener noreferrer";
      try {
        const parsedUrl = new URL(options.source);
        const hostClean = parsedUrl.hostname.replace(/^www\./, "");
        sourceLink.textContent = `Official source: ${hostClean}${parsedUrl.pathname.length > 1 ? parsedUrl.pathname : ""}`;
      } catch (e) {
        sourceLink.textContent = "Official DSU source";
      }
      bubble.appendChild(sourceLink);
    }

    // Optional Quick Replies
    if (options.quickReplies && options.quickReplies.length > 0) {
      const qrWrap = document.createElement("div");
      qrWrap.className = "quick-replies-wrap";
      options.quickReplies.forEach((replyText) => {
        const pill = document.createElement("button");
        pill.className = "quick-reply-pill";
        pill.textContent = replyText;
        pill.addEventListener("click", () => {
          sendMessage(replyText);
        });
        qrWrap.appendChild(pill);
      });
      bubble.appendChild(qrWrap);
    }

    row.appendChild(bubble);

    // Timestamp
    const timeSpan = document.createElement("div");
    timeSpan.className = "msg-timestamp";
    const now = new Date();
    timeSpan.textContent = now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    row.appendChild(timeSpan);

    messagesContainer.appendChild(row);

    // If show_enquiry_form is true, append interactive enquiry form card
    if (options.showEnquiryForm) {
      const enquiryCard = createEnquiryCard(options.formPrefill || {});
      messagesContainer.appendChild(enquiryCard);
      setTimeout(() => {
        scrollToBottom();
      }, 60);
    }
  }

  // 6. Interactive Inline Enquiry Form Card
  function createEnquiryCard(prefill = {}) {
    const card = document.createElement("div");
    card.className = "enquiry-card-inline";

    const prefillCourse = prefill.course_name || "";
    const prefillSchool = prefill.school_or_department || "";
    const prefillQuestion = prefill.question || "";

    card.innerHTML = `
      <div class="enquiry-card-header">
        <h5>🎓 DSU Admissions Query Submission</h5>
        <span style="font-size: 10px; color: #64748b; font-weight: 600;">Admissions Cell</span>
      </div>
      <form class="enquiry-card-form" id="enquiry-form-${Date.now()}">
        <div class="enquiry-form-group">
          <label>Full Name <span class="required">*</span></label>
          <input type="text" name="student_name" placeholder="Your Full Name" required />
        </div>
        <div class="enquiry-form-group">
          <label>Email Address <span class="required">*</span></label>
          <input type="email" name="email" placeholder="student@example.com" required />
        </div>
        <div class="enquiry-form-group">
          <label>Phone Number (Optional)</label>
          <input type="tel" name="phone_number" placeholder="10-digit mobile number" />
        </div>
        <div class="enquiry-form-group">
          <label>School / Department</label>
          <input type="text" name="school_or_department" value="${escapeHtml(prefillSchool)}" placeholder="e.g. School of Computer Applications" />
        </div>
        <div class="enquiry-form-group">
          <label>Programme of Interest</label>
          <input type="text" name="course_name" value="${escapeHtml(prefillCourse)}" placeholder="e.g. Bachelor of Computer Applications (BCA)" />
        </div>
        <div class="enquiry-form-group">
          <label>Your Question / Query <span class="required">*</span></label>
          <textarea name="question" placeholder="Please specify your query..." required>${escapeHtml(prefillQuestion)}</textarea>
        </div>
        <div class="enquiry-error-msg"></div>
        <div class="enquiry-form-actions">
          <button type="submit" class="enquiry-submit-btn">
            Submit Enquiry
          </button>
        </div>
        <div class="enquiry-success-box"></div>
      </form>
    `;

    const form = card.querySelector("form");
    const errorMsg = card.querySelector(".enquiry-error-msg");
    const successBox = card.querySelector(".enquiry-success-box");
    const submitBtn = card.querySelector(".enquiry-submit-btn");

    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      errorMsg.style.display = "none";
      errorMsg.textContent = "";

      const formData = new FormData(form);
      const studentName = formData.get("student_name")?.toString().trim();
      const email = formData.get("email")?.toString().trim();
      const phone = formData.get("phone_number")?.toString().trim() || null;
      const school = formData.get("school_or_department")?.toString().trim() || null;
      const course = formData.get("course_name")?.toString().trim() || null;
      const question = formData.get("question")?.toString().trim();

      // Client validation
      if (!studentName || !email || !question) {
        errorMsg.textContent = "Please fill in all required fields (Name, Email, Question).";
        errorMsg.style.display = "block";
        return;
      }

      const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
      if (!emailRegex.test(email)) {
        errorMsg.textContent = "Please provide a valid email address.";
        errorMsg.style.display = "block";
        return;
      }

      submitBtn.disabled = true;
      submitBtn.textContent = "Submitting...";

      try {
        const resp = await fetch(`${API_BASE}/enquiries`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            student_name: studentName,
            email: email,
            phone_number: phone,
            school_or_department: school,
            course_name: course,
            question: question,
            session_id: sessionId,
          }),
        });

        if (resp.status === 201) {
          const resData = await resp.json();
          form.querySelectorAll(".enquiry-form-group, .enquiry-form-actions").forEach((el) => {
            el.style.display = "none";
          });
          successBox.innerHTML = `
            <strong>Enquiry Successfully Registered!</strong><br />
            Thank you, ${escapeHtml(studentName)}. The DSU Admissions Cell has received your enquiry.
            <span class="confirm-id">Application Ref: #DSU-${resData.id}</span>
          `;
          successBox.style.display = "block";
          scrollToBottom();
        } else if (resp.status === 503) {
          errorMsg.textContent = "The database service is currently offline. Please contact admissions@dsu.edu.in directly.";
          errorMsg.style.display = "block";
          submitBtn.disabled = false;
          submitBtn.textContent = "Retry Submission";
        } else {
          const errBody = await resp.json().catch(() => ({}));
          errorMsg.textContent = errBody.detail || "Submission failed. Please check your details and try again.";
          errorMsg.style.display = "block";
          submitBtn.disabled = false;
          submitBtn.textContent = "Submit Enquiry";
        }
      } catch (err) {
        errorMsg.textContent = "Network error: Unable to submit enquiry to server.";
        errorMsg.style.display = "block";
        submitBtn.disabled = false;
        submitBtn.textContent = "Submit Enquiry";
      }
    });

    return card;
  }

  // 7. Helpers
  function createTypingIndicator() {
    const row = document.createElement("div");
    row.className = "chat-msg-row bot typing-indicator-row";
    row.innerHTML = `
      <div class="typing-bubble">
        <span class="typing-dot"></span>
        <span class="typing-dot"></span>
        <span class="typing-dot"></span>
      </div>
    `;
    return row;
  }

  function scrollToBottom() {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }

  function formatMarkdown(text) {
    if (!text) return "";

    // Normalize unicode spaces, hyphens, and remove raw HTML entities
    let cleaned = String(text)
      .replace(/\\r\\n/g, "\n")
      .replace(/\\n/g, "\n")
      .replace(/\\r/g, "")
      .replace(/&#x20;/gi, " ")
      .replace(/&nbsp;/gi, " ")
      .replace(/\\([#*_`[\]()\-+!<>])/g, "$1");

    let safe = escapeHtml(cleaned);

    // Bold: **Text**
    safe = safe.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");

    // Markdown Links: [Text](URL) -> sanitized clickable link
    safe = safe.replace(
      /\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
      '<a href="$2" target="_blank" rel="noopener noreferrer" class="chat-external-link">$1</a>'
    );

    // Convert bullet points and numbered lists into clean <ul> / <ol> lists
    const lines = safe.split("\n");
    let inList = false;
    let inNumList = false;
    const out = [];

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const trimmed = line.trim();

      if (/^[-*•]\s+(.*)$/.test(trimmed)) {
        if (inNumList) {
          out.push("</ol>");
          inNumList = false;
        }
        const content = trimmed.replace(/^[-*•]\s+/, "").trim();
        if (content.length > 0) {
          if (!inList) {
            out.push('<ul class="chat-bullet-list">');
            inList = true;
          }
          out.push(`<li>${content}</li>`);
        }
      } else if (/^\d+\.\s+(.*)$/.test(trimmed)) {
        if (inList) {
          out.push("</ul>");
          inList = false;
        }
        const numContent = trimmed.replace(/^\d+\.\s+/, "").trim();
        if (numContent.length > 0) {
          if (!inNumList) {
            out.push('<ol class="chat-numbered-list">');
            inNumList = true;
          }
          out.push(`<li>${numContent}</li>`);
        }
      } else {
        if (inList) {
          out.push("</ul>");
          inList = false;
        }
        if (inNumList) {
          out.push("</ol>");
          inNumList = false;
        }
        if (trimmed.length > 0) {
          out.push(line);
        } else if (out.length > 0 && out[out.length - 1] !== "") {
          out.push("");
        }
      }
    }
    if (inList) {
      out.push("</ul>");
    }
    if (inNumList) {
      out.push("</ol>");
    }

    let html = out.join("\n");
    html = html.replace(/\n/g, "<br />");
    html = html.replace(/<br \/><ul/g, "<ul");
    html = html.replace(/<\/ul><br \/>/g, "</ul>");
    html = html.replace(/<br \/><ol/g, "<ol");
    html = html.replace(/<\/ol><br \/>/g, "</ol>");
    html = html.replace(/(<br \/>\s*){3,}/g, "<br /><br />");
    html = html.replace(/^(?:<br \/>)+|(?:<br \/>)+$/g, "");
    return html;
  }

  function escapeHtml(string) {
    if (!string) return "";
    const entityMap = {
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    };
    return String(string).replace(/[&<>"']/g, (s) => entityMap[s]);
  }
}
