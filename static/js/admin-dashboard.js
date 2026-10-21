/**
 * DSU Admission Cell Enquiry Management Dashboard Script
 * Manages secure API interaction, dynamic search/filter, and persistent audit trail rendering.
 */

(function () {
    const API_BASE = window.location.port === '8000' ? '' : 'http://127.0.0.1:8000';
    const LOGIN_URL = window.location.port === '8000' ? '/admin/login' : '/';

    let currentStaffUser = null;
    let activeEnquiryId = null;
    let searchDebounceTimer = null;
    let autoRefreshTimer = null;

    // Helper: Auth headers
    function getAuthHeaders() {
        const token = sessionStorage.getItem('dsu_admin_token');
        const headers = {
            'Content-Type': 'application/json'
        };
        if (token) {
            headers['Authorization'] = `Bearer ${token}`;
        }
        return headers;
    }

    // Authenticate staff on load
    async function checkAuth() {
        try {
            const resp = await fetch(`${API_BASE}/admin/api/me`, {
                headers: getAuthHeaders(),
                credentials: 'include'
            });

            if (!resp.ok) {
                sessionStorage.removeItem('dsu_admin_token');
                sessionStorage.removeItem('dsu_admin_user');
                window.location.href = LOGIN_URL;
                return false;
            }

            currentStaffUser = await resp.json();
            document.getElementById('navStaffName').textContent = currentStaffUser.full_name || currentStaffUser.username;
            document.getElementById('navStaffRole').textContent = (currentStaffUser.role || 'staff').replace('_', ' ');
            return true;
        } catch (err) {
            console.error('Auth verification error:', err);
            window.location.href = LOGIN_URL;
            return false;
        }
    }

    // Format date string nicely
    function formatDate(isoStr) {
        if (!isoStr) return '-';
        try {
            const d = new Date(isoStr);
            return d.toLocaleString('en-IN', {
                day: '2-digit',
                month: 'short',
                year: 'numeric',
                hour: '2-digit',
                minute: '2-digit'
            });
        } catch {
            return isoStr;
        }
    }

    // Escape HTML to prevent XSS
    function escapeHtml(str) {
        if (!str) return '';
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    // Load and render enquiries list with auto-refresh support
    async function loadEnquiries(isAutoRefresh = false) {
        const tbody = document.getElementById('enquiriesTableBody');
        const searchInput = document.getElementById('searchInput');
        const search = searchInput ? searchInput.value.trim() : '';
        const status = document.getElementById('statusFilter') ? document.getElementById('statusFilter').value : '';
        const school = document.getElementById('schoolFilter') ? document.getElementById('schoolFilter').value : '';

        const params = new URLSearchParams();
        if (search) params.append('search', search);
        if (status) params.append('status', status);
        if (school) params.append('school', school);

        if (!isAutoRefresh && (!tbody.children || tbody.children.length === 0)) {
            tbody.innerHTML = '<tr><td colspan="7" class="empty-state">Loading enquiries...</td></tr>';
        }

        try {
            const resp = await fetch(`${API_BASE}/admin/api/enquiries?${params.toString()}`, {
                headers: getAuthHeaders(),
                credentials: 'include'
            });

            if (!resp.ok) {
                if (resp.status === 401) {
                    window.location.href = LOGIN_URL;
                    return;
                }
                throw new Error('Failed to load enquiries');
            }

            const data = await resp.json();
            renderTable(data.items);
            updateMetrics(data.total, data.status_counts);
        } catch (err) {
            if (!isAutoRefresh) {
                console.error('Error loading enquiries:', err);
                tbody.innerHTML = `<tr><td colspan="7" class="empty-state" style="color: #ef4444;">Failed to load data. Please refresh.</td></tr>`;
            }
        }
    }

    function updateMetrics(total, counts) {
        counts = counts || {};
        document.getElementById('statTotal').textContent = total || 0;
        document.getElementById('totalBadge').textContent = `${total || 0} records`;

        const proc = (counts['processing'] || 0) + (counts['new'] || 0);
        const called = counts['called'] || 0;
        const visited = counts['visited'] || 0;
        const resolved = (counts['resolved'] || 0) + (counts['closed'] || 0);

        document.getElementById('statProcessing').textContent = proc;
        document.getElementById('statCalled').textContent = called;
        document.getElementById('statVisited').textContent = visited;
        document.getElementById('statResolved').textContent = resolved;
    }

    function renderTable(items) {
        const tbody = document.getElementById('enquiriesTableBody');
        if (!items || items.length === 0) {
            tbody.innerHTML = '<tr><td colspan="7" class="empty-state">No admission enquiries match the current filters.</td></tr>';
            return;
        }

        tbody.innerHTML = items.map(item => {
            const statusClass = (item.status || 'new').toLowerCase().replace(' ', '_');
            const phoneDisplay = item.phone_number ? `<br>📞 ${escapeHtml(item.phone_number)}` : '';
            const schoolDisplay = item.school_or_department ? `<small style="color:#64748b;">${escapeHtml(item.school_or_department)}</small><br>` : '';

            return `
                <tr>
                    <td><strong>#${item.id}</strong></td>
                    <td style="white-space: nowrap; font-size: 12px; color: #475569;">${formatDate(item.created_at)}</td>
                    <td>
                        <div class="student-name">${escapeHtml(item.student_name)}</div>
                        <div class="student-contact">✉ ${escapeHtml(item.email)}${phoneDisplay}</div>
                    </td>
                    <td>
                        ${schoolDisplay}
                        <strong>${escapeHtml(item.course_name || 'General Admission')}</strong>
                    </td>
                    <td>
                        <div class="question-snippet" title="${escapeHtml(item.question)}">${escapeHtml(item.question)}</div>
                    </td>
                    <td>
                        <span class="status-pill ${statusClass}">${escapeHtml(item.status)}</span>
                    </td>
                    <td>
                        <button class="btn-manage" data-enquiry-id="${item.id}">Review & Manage</button>
                    </td>
                </tr>
            `;
        }).join('');

        // Attach click listeners to manage buttons
        tbody.querySelectorAll('.btn-manage').forEach(btn => {
            btn.addEventListener('click', () => {
                const id = btn.getAttribute('data-enquiry-id');
                openCaseModal(id);
            });
        });
    }

    // Modal & Case Review
    async function openCaseModal(enquiryId) {
        activeEnquiryId = enquiryId;
        const modal = document.getElementById('caseModal');

        try {
            const resp = await fetch(`${API_BASE}/admin/api/enquiries/${enquiryId}`, {
                headers: getAuthHeaders(),
                credentials: 'include'
            });

            if (!resp.ok) {
                alert('Could not retrieve enquiry details.');
                return;
            }

            const detail = await resp.json();
            populateModal(detail);
            modal.style.display = 'flex';
        } catch (err) {
            console.error('Error fetching case detail:', err);
            alert('Error loading enquiry case.');
        }
    }

    function populateModal(detail) {
        document.getElementById('modalCaseId').textContent = `Enquiry Case #${detail.id}`;

        const statusClass = (detail.status || 'new').toLowerCase().replace(' ', '_');
        const badge = document.getElementById('modalStatusBadge');
        badge.className = `status-pill ${statusClass}`;
        badge.textContent = detail.status;

        document.getElementById('modalStudentName').textContent = detail.student_name || '-';
        document.getElementById('modalEmail').textContent = detail.email || '-';
        document.getElementById('modalPhone').textContent = detail.phone_number || 'Not provided';
        document.getElementById('modalSchool').textContent = detail.school_or_department || 'General';
        document.getElementById('modalCourse').textContent = detail.course_name || 'Undergraduate / Postgraduate';
        document.getElementById('modalSubmittedAt').textContent = formatDate(detail.created_at);
        document.getElementById('modalQuestion').textContent = detail.question || '-';

        // Set status dropdown
        const statusSelect = document.getElementById('newStatusSelect');
        statusSelect.value = detail.status.toLowerCase();
        document.getElementById('statusNoteInput').value = '';

        // Reset comment and follow-up inputs
        document.getElementById('commentTextInput').value = '';
        document.getElementById('followUpNotes').value = '';

        // Render audit timeline
        renderTimeline(detail.timeline);
    }

    function renderTimeline(events) {
        const container = document.getElementById('auditTimeline');
        if (!events || events.length === 0) {
            container.innerHTML = '<div style="color: #64748b; font-size: 13px;">No audit events recorded.</div>';
            return;
        }

        // Render in chronological order
        container.innerHTML = events.map(evt => {
            const itemClass = evt.type || 'comment';
            return `
                <div class="timeline-item ${itemClass}">
                    <div class="timeline-marker"></div>
                    <div class="timeline-content">
                        <div class="timeline-meta">
                            <span class="timeline-actor">👤 ${escapeHtml(evt.actor || 'System')}</span>
                            <span>${formatDate(evt.timestamp)}</span>
                        </div>
                        <div class="timeline-action">${escapeHtml(evt.title || evt.action || 'Audit Event')}</div>
                        ${evt.details ? `<div class="timeline-details">${escapeHtml(evt.details)}</div>` : ''}
                    </div>
                </div>
            `;
        }).join('');
    }

    function closeModal() {
        document.getElementById('caseModal').style.display = 'none';
        activeEnquiryId = null;
    }

    // Forms Handlers
    async function handleStatusUpdate(e) {
        e.preventDefault();
        if (!activeEnquiryId) return;

        const btn = document.getElementById('btnUpdateStatus');
        const newStatus = document.getElementById('newStatusSelect').value;
        const notes = document.getElementById('statusNoteInput').value.trim();

        btn.disabled = true;
        btn.textContent = 'Updating...';

        try {
            const resp = await fetch(`${API_BASE}/admin/api/enquiries/${activeEnquiryId}/status`, {
                method: 'PATCH',
                headers: getAuthHeaders(),
                credentials: 'include',
                body: JSON.stringify({ status: newStatus, notes: notes || null })
            });

            if (!resp.ok) {
                const err = await resp.json();
                throw new Error(err.detail || 'Failed to update status');
            }

            const updated = await resp.json();
            populateModal(updated);
            loadEnquiries(); // Refresh background list
        } catch (err) {
            alert('Error updating status: ' + err.message);
        } finally {
            btn.disabled = false;
            btn.textContent = 'Update Status';
        }
    }

    async function handleAddComment(e) {
        e.preventDefault();
        if (!activeEnquiryId) return;

        const btn = document.getElementById('btnAddComment');
        const commentText = document.getElementById('commentTextInput').value.trim();
        if (!commentText) return;

        btn.disabled = true;
        btn.textContent = 'Posting...';

        try {
            const resp = await fetch(`${API_BASE}/admin/api/enquiries/${activeEnquiryId}/comments`, {
                method: 'POST',
                headers: getAuthHeaders(),
                credentials: 'include',
                body: JSON.stringify({ comment_text: commentText })
            });

            if (!resp.ok) {
                const err = await resp.json();
                throw new Error(err.detail || 'Failed to add comment');
            }

            const updated = await resp.json();
            populateModal(updated);
            document.getElementById('commentTextInput').value = '';
        } catch (err) {
            alert('Error adding comment: ' + err.message);
        } finally {
            btn.disabled = false;
            btn.textContent = 'Post Comment';
        }
    }

    async function handleRecordFollowUp(e) {
        e.preventDefault();
        if (!activeEnquiryId) return;

        const btn = document.getElementById('btnRecordFollowUp');
        const actionType = document.getElementById('followUpType').value;
        const notes = document.getElementById('followUpNotes').value.trim();
        if (!notes) return;

        btn.disabled = true;
        btn.textContent = 'Recording...';

        try {
            const resp = await fetch(`${API_BASE}/admin/api/enquiries/${activeEnquiryId}/follow-ups`, {
                method: 'POST',
                headers: getAuthHeaders(),
                credentials: 'include',
                body: JSON.stringify({ action_type: actionType, notes: notes })
            });

            if (!resp.ok) {
                const err = await resp.json();
                throw new Error(err.detail || 'Failed to record follow-up');
            }

            const updated = await resp.json();
            populateModal(updated);
            document.getElementById('followUpNotes').value = '';
        } catch (err) {
            alert('Error recording follow-up: ' + err.message);
        } finally {
            btn.disabled = false;
            btn.textContent = 'Record Action';
        }
    }

    async function handleLogout() {
        try {
            await fetch(`${API_BASE}/admin/api/logout`, {
                method: 'POST',
                headers: getAuthHeaders(),
                credentials: 'include'
            });
        } catch (e) {
            // ignore
        }
        if (autoRefreshTimer) clearInterval(autoRefreshTimer);
        sessionStorage.removeItem('dsu_admin_token');
        sessionStorage.removeItem('dsu_admin_user');
        window.location.href = LOGIN_URL;
    }

    function startAutoRefresh() {
        if (autoRefreshTimer) clearInterval(autoRefreshTimer);
        autoRefreshTimer = setInterval(() => {
            const searchInput = document.getElementById('searchInput');
            if (!document.hidden && (!searchInput || document.activeElement !== searchInput)) {
                loadEnquiries(true);
            }
        }, 2500);
    }

    // Initialize Event Listeners
    document.addEventListener('DOMContentLoaded', async () => {
        const authed = await checkAuth();
        if (!authed) return;

        loadEnquiries();
        startAutoRefresh();

        // Filters
        document.getElementById('searchInput').addEventListener('input', () => {
            clearTimeout(searchDebounceTimer);
            searchDebounceTimer = setTimeout(loadEnquiries, 250);
        });
        document.getElementById('statusFilter').addEventListener('change', loadEnquiries);
        document.getElementById('schoolFilter').addEventListener('change', loadEnquiries);
        document.getElementById('refreshBtn').addEventListener('click', () => loadEnquiries(false));

        // Logout
        document.getElementById('logoutBtn').addEventListener('click', handleLogout);

        // Modal
        document.getElementById('modalCloseBtn').addEventListener('click', closeModal);
        document.getElementById('caseModal').addEventListener('click', (e) => {
            if (e.target.id === 'caseModal') closeModal();
        });

        // Form Submissions
        document.getElementById('statusUpdateForm').addEventListener('submit', handleStatusUpdate);
        document.getElementById('addCommentForm').addEventListener('submit', handleAddComment);
        document.getElementById('followUpForm').addEventListener('submit', handleRecordFollowUp);
    });
})();
