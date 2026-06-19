/* ===== ScholarMind Frontend App ===== */
(function () {
  'use strict';

  const API = '/api/v1';

  // ----- State -----
  const state = {
    userId: null,
    sessions: [],     // [{id, title, messages[], createdAt}]
    activeSessionId: null,
    selectedFeature: '',
    isLoading: false,
    sidebarOpen: true,
    adminMode: false,
    abortController: null,
    lowScorePage: 0,
    lowScorePageSize: 20,
    lowScoreSortKey: 'created_at',
    lowScoreSortDir: 'desc',   // default: most recent first
    lowScoreData: [],
    lowScoreSearch: '',
    lowScoreTypeFilter: '',
    adminTab: 'dashboard',
    requestMode: 'all',
  };

  const MAX_MESSAGES_PER_SESSION = 100;
  const STORAGE_QUOTA_MB = 4;  // stay under 4MB to avoid 5MB browser limit
  const PERSISTENT_SESSION_FEATURES = new Set();
  const FEATURES = {
    '': { label: 'Chat', endpoint: '/chat/stream' },
  };

  // ----- DOM refs -----
  const $ = (s) => document.querySelector(s);
  const $$ = (s) => document.querySelectorAll(s);

  // ----- Helpers -----
  async function apiFetch(url, options) {
    const resp = await fetch(url, options);
    if (!resp.ok) {
      const text = await resp.text();
      let msg;
      try { msg = JSON.parse(text).error?.message || JSON.parse(text).detail || resp.statusText; }
      catch { msg = text.substring(0, 100); }
      throw new Error(`HTTP ${resp.status}: ${msg}`);
    }
    return resp.json();
  }

  async function apiFetchOptional(url, options, allowedStatuses) {
    try {
      return await apiFetch(url, options);
    } catch (err) {
      const statuses = allowedStatuses || [404];
      if (err instanceof Error && statuses.some((status) => err.message.startsWith(`HTTP ${status}:`))) {
        return null;
      }
      throw err;
    }
  }

  // ----- Init -----
  document.addEventListener('DOMContentLoaded', () => {
    if (typeof marked !== 'undefined') {
      marked.setOptions({ breaks: true, gfm: true });
    }

    // Redraw admin charts on window resize
    let resizeTimer;
    window.addEventListener('resize', () => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => {
        if (state.adminMode && state._lastTrendResp) {
          renderScoreTrend(state._lastTrendResp);
          if (state._lastOnlineResp) {
            renderByReportType(state._lastOnlineResp);
          }
        }
      }, 200);
    });

    const savedUser = localStorage.getItem('sm_user_id');
    if (savedUser) {
      state.userId = savedUser;
      loadSessions();
      showApp();
    }
    bindEvents();
  });

  // ----- Event Bindings -----
  function bindEvents() {
    $('#login-btn').addEventListener('click', handleLogin);
    $('#login-user-id').addEventListener('keydown', (e) => {
      if (e.key === 'Enter') handleLogin();
    });
    $('#new-chat-btn').addEventListener('click', createNewChat);
    $('#send-btn').addEventListener('click', sendMessage);
    $('#message-input').addEventListener('keydown', handleInputKeydown);
    $('#message-input').addEventListener('input', autoResize);
    $('#sidebar-toggle').addEventListener('click', toggleSidebar);
    $('#user-info').addEventListener('click', handleLogout);
    $('#feature-btn').addEventListener('click', toggleFeatureDropdown);
    $('#feature-label-close').addEventListener('click', clearFeature);
    $('#admin-back-btn').addEventListener('click', () => switchToChat());
    $('#admin-refresh').addEventListener('click', loadAdminDashboard);
    $('#admin-time-range').addEventListener('change', loadAdminDashboard);
    $('#admin-auto-refresh').addEventListener('change', toggleAutoRefresh);

    // Admin sidebar tabs
    $$('.admin-tab').forEach(tab => {
      tab.addEventListener('click', () => showAdminTab(tab.dataset.tab));
    });

    // Requests mode bar
    $$('.mode-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        state.requestMode = btn.dataset.mode;
        state.lowScorePage = 0;
        $$('.mode-btn').forEach(b => b.classList.toggle('active', b.dataset.mode === state.requestMode));
        loadRequestsPanel();
      });
    });

    $('#ls-prev').addEventListener('click', () => { state.lowScorePage = Math.max(0, state.lowScorePage - 1); loadRequestsPanel(); });
    $('#ls-next').addEventListener('click', () => { state.lowScorePage++; loadRequestsPanel(); });

    // Low-score table sort headers
    $$('.low-scores-section th.sortable').forEach((th) => {
      th.addEventListener('click', () => {
        const key = th.dataset.sort;
        if (state.lowScoreSortKey === key) {
          state.lowScoreSortDir = state.lowScoreSortDir === 'asc' ? 'desc' : 'asc';
        } else {
          state.lowScoreSortKey = key;
          state.lowScoreSortDir = key === 'created_at' ? 'desc' : 'asc';
        }
        applyLowScoreFilters();
      });
    });

    // Low-score search and type filter
    const lsSearch = $('#ls-search');
    let lsSearchTimer;
    lsSearch.addEventListener('input', () => {
      clearTimeout(lsSearchTimer);
      lsSearchTimer = setTimeout(() => {
        state.lowScoreSearch = lsSearch.value.trim().toLowerCase();
        state.lowScorePage = 0;
        applyLowScoreFilters();
      }, 300);
    });
    $('#ls-type-filter').addEventListener('change', (e) => {
      state.lowScoreTypeFilter = e.target.value;
      state.lowScorePage = 0;
      applyLowScoreFilters();
    });

    const userFilter = $('#admin-user-filter');
    let userFilterTimer;
    userFilter.addEventListener('input', () => {
      clearTimeout(userFilterTimer);
      userFilterTimer = setTimeout(() => loadAdminDashboard(), 500);
    });

    $('#admin-export').addEventListener('click', exportDashboardCsv);

    // Feature dropdown options
    $$('.feature-option').forEach((el) => {
      el.addEventListener('click', () => selectFeature(el.dataset.value));
    });

    // Theme toggle
    const themeToggle = $('#theme-toggle');
    if (themeToggle) {
      themeToggle.addEventListener('click', () => {
        const html = document.documentElement;
        const isLight = html.getAttribute('data-theme') === 'light';
        if (isLight) {
          html.removeAttribute('data-theme');
          localStorage.setItem('sm_theme', 'dark');
        } else {
          html.setAttribute('data-theme', 'light');
          localStorage.setItem('sm_theme', 'light');
        }
        // Redraw admin charts if in admin mode
        if (state.adminMode && state._lastTrendResp) {
          renderScoreTrend(state._lastTrendResp);
          if (state._lastOnlineResp) {
            renderByReportType(state._lastOnlineResp);
          }
        }
      });
    }

    // Close dropdown on outside click
    document.addEventListener('click', (e) => {
      if (!e.target.closest('.feature-selector')) {
        $('#feature-dropdown').classList.add('hidden');
      }
    });
  }

  // ----- Stream Abort -----
  function abortActiveStream() {
    if (state.abortController) {
      state.abortController.abort();
      state.abortController = null;
    }
  }

  // ----- Auth -----
  function handleLogin() {
    const input = $('#login-user-id');
    const userId = input.value.trim();
    if (!userId) return;
    state.userId = userId;
    localStorage.setItem('sm_user_id', userId);
    loadSessions();
    showApp();
  }

  function handleLogout() {
    if (!confirm('Log out?')) return;
    abortActiveStream();
    localStorage.removeItem('sm_user_id');
    localStorage.removeItem('sm_sessions_' + state.userId);
    state.userId = null;
    state.sessions = [];
    state.activeSessionId = null;
    state.selectedFeature = '';
    state.adminMode = false;
    $('#login-modal').classList.remove('hidden');
    $('#app').classList.add('hidden');
    $('#login-user-id').value = '';
  }

  function showApp() {
    $('#login-modal').classList.add('hidden');
    $('#app').classList.remove('hidden');
    $('#user-name').textContent = state.userId;
    $('#user-avatar').textContent = state.userId[0].toUpperCase();

    if (state.userId === 'admin') {
      addAdminNav();
    } else {
      removeAdminNav();
    }
    syncFeatureForActiveSession();
    renderSidebar();
    renderChat();
  }

  // ----- Admin Nav -----
  function addAdminNav() {
    if ($('.nav-admin')) return;
    const sep = document.createElement('div');
    sep.className = 'sidebar-sep';
    const nav = document.createElement('div');
    nav.className = 'nav-admin';
    nav.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg> Dashboard`;
    nav.addEventListener('click', () => switchToAdmin());
    const list = $('#conversation-list');
    list.parentNode.insertBefore(sep, list.nextSibling);
    list.parentNode.insertBefore(nav, sep.nextSibling);
  }

  function removeAdminNav() {
    const nav = $('.nav-admin');
    const sep = nav?.previousElementSibling;
    if (sep?.classList.contains('sidebar-sep')) sep.remove();
    nav?.remove();
  }

  // ----- Sessions -----
  function loadSessions() {
    try {
      const raw = localStorage.getItem('sm_sessions_' + state.userId);
      state.sessions = raw ? JSON.parse(raw) : [];
    } catch {
      state.sessions = [];
    }
    if (state.sessions.length > 0 && !state.activeSessionId) {
      state.activeSessionId = state.sessions[0].id;
    }
  }

  function saveSessions() {
    // Trim old messages per session to prevent unbounded growth
    for (const s of state.sessions) {
      if (s.messages.length > MAX_MESSAGES_PER_SESSION) {
        s.messages = s.messages.slice(-MAX_MESSAGES_PER_SESSION);
      }
    }
    try {
      const key = 'sm_sessions_' + state.userId;
      let data = JSON.stringify(state.sessions);
      // Check approximate size and trim further if needed
      let sizeMB = new Blob([data]).size / (1024 * 1024);
      if (sizeMB > STORAGE_QUOTA_MB) {
        for (let i = state.sessions.length - 1; i >= 0; i--) {
          const half = Math.floor(state.sessions[i].messages.length / 2);
          if (half > 0) state.sessions[i].messages = state.sessions[i].messages.slice(-half);
        }
        data = JSON.stringify(state.sessions);
      }
      localStorage.setItem(key, data);
    } catch (e) {
      console.warn('Failed to save sessions to localStorage:', e);
    }
  }

  function getActiveSession() {
    return state.sessions.find((s) => s.id === state.activeSessionId);
  }

  // ----- Create Chat -----
  async function createNewChat() {
    try {
      const json = await apiFetch(`${API}/sessions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: state.userId }),
      });
      if (!json.success) throw new Error(json.error?.message || 'Failed to create session');

      const session = {
        id: json.data.session_id,
        title: 'New Chat',
        messages: [],
        feature: '',
        createdAt: new Date().toISOString(),
      };
      state.sessions.unshift(session);
      state.activeSessionId = session.id;
      selectFeature('');
      saveSessions();
      renderSidebar();
      renderChat();
      $('#message-input').focus();
    } catch (err) {
      console.error('Create session error:', err);
      alert('Failed to create session: ' + err.message);
    }
  }

  function switchSession(sessionId) {
    abortActiveStream();
    state.activeSessionId = sessionId;
    syncFeatureForActiveSession();
    renderSidebar();
    renderChat();
  }

  function deleteSession(sessionId, e) {
    e.stopPropagation();
    abortActiveStream();
    const idx = state.sessions.findIndex((s) => s.id === sessionId);
    if (idx < 0) return;
    state.sessions.splice(idx, 1);
    if (state.activeSessionId === sessionId) {
      state.activeSessionId = state.sessions.length > 0 ? state.sessions[0].id : null;
    }
    syncFeatureForActiveSession();
    saveSessions();
    renderSidebar();
    renderChat();
    // Close on server
    fetch(`${API}/sessions/${sessionId}`, { method: 'DELETE' }).catch((err) => {
      console.warn('Failed to delete session on server:', err);
    });
  }

  // ----- Sidebar -----
  function renderSidebar() {
    const list = $('#conversation-list');
    list.innerHTML = '';
    state.sessions.forEach((s) => {
      const div = document.createElement('div');
      div.className = 'conv-item' + (s.id === state.activeSessionId ? ' active' : '');
      div.innerHTML = `
        <span class="conv-item-title">${escapeHtml(s.title)}</span>
        <button class="conv-item-delete" title="Delete">&times;</button>
      `;
      div.addEventListener('click', () => switchSession(s.id));
      div.querySelector('.conv-item-delete').addEventListener('click', (e) => deleteSession(s.id, e));
      list.appendChild(div);
    });
  }

  function toggleSidebar() {
    state.sidebarOpen = !state.sidebarOpen;
    $('#sidebar').classList.toggle('collapsed', !state.sidebarOpen);
  }

  // ----- Feature Selector -----
  function toggleFeatureDropdown() {
    $('#feature-dropdown').classList.toggle('hidden');
  }

  function selectFeature(value) {
    state.selectedFeature = value;
    $$('.feature-option').forEach((el) => {
      el.classList.toggle('selected', el.dataset.value === value);
    });
    $('#feature-dropdown').classList.add('hidden');
    const label = FEATURES[value]?.label || 'Memory Eval';
    $('#feature-btn').classList.toggle('active', !!value);

    if (value) {
      $('#feature-label').classList.remove('hidden');
      $('#feature-label-text').textContent = label;
    } else {
      $('#feature-label').classList.add('hidden');
    }
  }

  function clearFeature() {
    selectFeature('');
  }

  function shouldPersistFeature(value) {
    return PERSISTENT_SESSION_FEATURES.has(value);
  }

  function syncFeatureForActiveSession() {
    const session = getActiveSession();
    selectFeature(session?.feature || '');
  }

  // ----- Chat -----
  function renderChat() {
    const session = getActiveSession();
    const container = $('#chat-messages');

    if (!session) {
      container.innerHTML = '';
      container.appendChild(createWelcomeScreen());
      return;
    }

    if (session.messages.length === 0) {
      container.innerHTML = '';
      container.appendChild(createWelcomeScreen());
      return;
    }

    container.innerHTML = '';
    const frag = document.createDocumentFragment();
    session.messages.forEach((msg) => {
      frag.appendChild(createMessageElement(msg));
    });
    container.appendChild(frag);
    scrollToBottom();
  }

  function createWelcomeScreen() {
    const div = document.createElement('div');
    div.id = 'welcome-screen';
    div.className = 'welcome-screen';
    div.innerHTML = `
      <div class="welcome-icon">
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
          <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/>
          <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>
        </svg>
      </div>
      <h2>ScholarMind</h2>
      <p>Memory Evaluation Runtime</p>
      <div class="feature-cards">
        <div class="feature-card" data-feature=""><div class="feature-card-icon"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 3v18h18"/><path d="M7 15l3-3 3 2 4-6"/></svg></div><div class="feature-card-text">Dashboard</div></div>
        <div class="feature-card" data-feature=""><div class="feature-card-icon"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 11l3 3L22 4"/><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/></svg></div><div class="feature-card-text">Official Benchmark</div></div>
        <div class="feature-card" data-feature=""><div class="feature-card-icon"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg></div><div class="feature-card-text">Eval Export</div></div>
      </div>
    `;
    div.querySelectorAll('.feature-card').forEach((el) => {
      el.addEventListener('click', () => {
        selectFeature(el.dataset.feature);
        $('#message-input').focus();
      });
    });
    return div;
  }

  function createMessageElement(msg) {
    const div = document.createElement('div');
    div.className = `message ${msg.role}`;

    let bodyHtml = '';
    if (msg.role === 'user') {
      bodyHtml = `<p>${escapeHtml(msg.content)}</p>`;
    } else {
      bodyHtml = renderMarkdown(msg.content || '');
    }

    let citationsHtml = '';
    if (msg.citations && msg.citations.length > 0) {
      const uniqueCitations = [];
      const seenCitationKeys = new Set();
      msg.citations.forEach((citation) => {
        const key = citation.paper_id || citation.title || JSON.stringify(citation);
        if (seenCitationKeys.has(key)) return;
        seenCitationKeys.add(key);
        uniqueCitations.push(citation);
      });
      citationsHtml = '<div class="message-citations">' +
        uniqueCitations.map((c) => `<span class="citation-tag" title="${escapeHtml(c.title || '')}">${escapeHtml(c.paper_id || 'cite')}</span>`).join('') +
        '</div>';
    }

    const userAvatarSvg = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="8" r="4"/><path d="M20 21a8 8 0 1 0-16 0"/></svg>';
    const botAvatarSvg = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="6" y="8" width="12" height="10" rx="2"/><circle cx="10" cy="13" r="1" fill="currentColor"/><circle cx="14" cy="13" r="1" fill="currentColor"/><line x1="12" y1="4" x2="12" y2="8"/><circle cx="12" cy="3" r="1.5"/><line x1="4" y1="12" x2="6" y2="12"/><line x1="18" y1="12" x2="20" y2="12"/></svg>';
    div.innerHTML = `
      <div class="message-row">
        <div class="message-avatar">${msg.role === 'user' ? userAvatarSvg : botAvatarSvg}</div>
        <div class="message-body">${bodyHtml}${citationsHtml}</div>
      </div>
    `;
    return div;
  }

  function createStreamingMessage() {
    const div = document.createElement('div');
    div.className = 'message assistant';
    div.id = 'streaming-msg';
    const botAvatarSvg = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="6" y="8" width="12" height="10" rx="2"/><circle cx="10" cy="13" r="1" fill="currentColor"/><circle cx="14" cy="13" r="1" fill="currentColor"/><line x1="12" y1="4" x2="12" y2="8"/><circle cx="12" cy="3" r="1.5"/><line x1="4" y1="12" x2="6" y2="12"/><line x1="18" y1="12" x2="20" y2="12"/></svg>';
    div.innerHTML = `
      <div class="message-row">
        <div class="message-avatar">${botAvatarSvg}</div>
        <div class="message-body">
          <div class="typing-indicator"><span></span><span></span><span></span></div>
        </div>
      </div>
    `;
    return div;
  }

  function updateStreamingMessage(content) {
    const el = $('#streaming-msg .message-body');
    if (!el) return;
    el.innerHTML = renderMarkdown(content);
    scrollToBottom();
  }

  // ----- Send Message -----
  async function sendMessage() {
    const input = $('#message-input');
    const text = input.value.trim();
    if (!text || state.isLoading) return;

    // Ensure session exists
    if (!state.activeSessionId) {
      await createNewChat();
      if (!state.activeSessionId) return;
    }

    const session = getActiveSession();
    if (!session) return;

    // Add user message
    const userMsg = { role: 'user', content: text };
    session.messages.push(userMsg);
    const featureUsed = state.selectedFeature;
    session.feature = shouldPersistFeature(featureUsed) ? featureUsed : '';

    // Update title from first message
    if (session.messages.length === 1) {
      session.title = text.length > 30 ? text.substring(0, 30) + '...' : text;
      renderSidebar();
    }

    // Clear input
    input.value = '';
    autoResize.call(input);
    updateSendButton();

    // Remove welcome screen if present
    const welcome = $('#welcome-screen');
    if (welcome) welcome.remove();

    // Render user message
    const container = $('#chat-messages');
    container.appendChild(createMessageElement(userMsg));

    // Show streaming indicator
    const streamEl = createStreamingMessage();
    container.appendChild(streamEl);
    scrollToBottom();

    state.isLoading = true;
    updateSendButton();

    // Abort any previous in-flight stream
    abortActiveStream();
    state.abortController = new AbortController();

    try {
      const result = await streamRequest(text, session.id, state.abortController.signal, (chunk) => {
        updateStreamingMessage(chunk);
      });

      const assistantMsg = {
        role: 'assistant',
        content: result.answer || '',
        citations: result.citations || [],
      };
      session.messages.push(assistantMsg);
      saveSessions();

      // Replace streaming element
      streamEl.remove();
      container.appendChild(createMessageElement(assistantMsg));
      scrollToBottom();
    } catch (err) {
      console.error('Stream error:', err);
      if (err.name !== 'AbortError') {
        streamEl.remove();
        const errMsg = {
          role: 'assistant',
          content: `**Error:** ${err.message}. Please try again.`,
        };
        session.messages.push(errMsg);
        saveSessions();
        container.appendChild(createMessageElement(errMsg));
      }
    } finally {
      state.isLoading = false;
      state.abortController = null;
      if (!shouldPersistFeature(featureUsed)) {
        selectFeature('');
      }
      updateSendButton();
    }
  }

  // ----- Stream Request -----
  async function streamRequest(text, sessionId, signal, onChunk) {
    const feature = state.selectedFeature;
    const featureCfg = FEATURES[feature] || FEATURES[''];
    if (!featureCfg.endpoint) {
      const answer = (
        'No chat endpoint is configured.'
      );
      onChunk(answer);
      return { answer, citations: [] };
    }
    const endpoint = `${API}${featureCfg.endpoint}`;
    let fullText = '';
    let citations = [];

    const body = buildRequestBody(feature, text, sessionId);
    const resp = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal,
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.error?.message || `HTTP ${resp.status}`);
    }

    const contentType = resp.headers.get('content-type') || '';

    if (contentType.includes('text/event-stream')) {
      // SSE
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let currentEvent = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('event: ')) {
            currentEvent = line.slice(7).trim();
          } else if (line.startsWith('data: ')) {
            const data = line.slice(6).trim();
            if (data === '[DONE]') { currentEvent = ''; continue; }
            if (currentEvent === 'error') {
              currentEvent = '';
              let errMsg = data;
              try { const p = JSON.parse(data); errMsg = p.message || p.error || data; } catch {}
              throw new Error(errMsg || 'Stream error from server');
            }
            currentEvent = '';
            try {
              const parsed = JSON.parse(data);
              if (parsed.answer) {
                // Final answer from reviewer — replace accumulated drafts
                fullText = parsed.answer;
                onChunk(fullText);
              } else if (parsed.content || parsed.text) {
                // Intermediate chunk (draft / progress text)
                fullText += (parsed.content || parsed.text);
                onChunk(fullText);
              }
              if (parsed.citations) citations = parsed.citations;
              if (parsed.data) {
                if (parsed.data.answer) {
                  fullText = parsed.data.answer;
                  onChunk(fullText);
                }
                if (parsed.data.citations) citations = parsed.data.citations;
              }
            } catch (e) {
              if (e.message && !e.message.includes('JSON')) throw e;
              // Not JSON, treat as plain text
              fullText += data;
              onChunk(fullText);
            }
          }
        }
      }
    } else {
      // JSON response (non-streaming fallback)
      const json = await resp.json();
      if (json.success && json.data) {
        fullText = json.data.answer || json.data.content || JSON.stringify(json.data, null, 2);
        citations = json.data.citations || [];
      } else if (json.answer) {
        fullText = json.answer;
      } else {
        fullText = JSON.stringify(json, null, 2);
      }
      onChunk(fullText);
    }

    return { answer: fullText, citations };
  }

  function buildRequestBody(feature, text, sessionId) {
    const base = {
      user_id: state.userId,
      session_id: sessionId,
    };
    return { ...base, query: text };
  }

  // ----- Input Helpers -----
  function handleInputKeydown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  function autoResize() {
    const el = this instanceof HTMLTextAreaElement ? this : $('#message-input');
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 200) + 'px';
    updateSendButton();
  }

  function updateSendButton() {
    const btn = $('#send-btn');
    const hasText = $('#message-input').value.trim().length > 0;
    btn.disabled = !hasText || state.isLoading;
    btn.classList.toggle('loading', state.isLoading);
  }

  function scrollToBottom() {
    const container = $('#chat-messages');
    requestAnimationFrame(() => {
      container.scrollTop = container.scrollHeight;
    });
  }

  // ----- Admin Dashboard -----
  const TREND_LINES = [
    { key: 'avg_overall_score',        label: 'Overall',  colorDark: '#00e5a0', colorLight: '#00b882', active: true },
    { key: 'avg_memory_score',         label: 'Memory',   colorDark: '#f59e0b', colorLight: '#d97706', active: true },
    { key: 'avg_answer_quality_score', label: 'Answer',   colorDark: '#a855f7', colorLight: '#7c3aed', active: true },
  ];

  function trendLineColor(line) {
    const light = document.documentElement.getAttribute('data-theme') === 'light';
    return light ? line.colorLight : line.colorDark;
  }

  function switchToAdmin() {
    state.adminMode = true;
    state.adminTab = 'dashboard';
    $('#sidebar-chat').classList.add('hidden');
    $('#sidebar-admin').classList.remove('hidden');
    $('#chat-view').classList.add('hidden');
    $('#admin-view').classList.remove('hidden');
    showAdminTab('dashboard');
    loadAdminDashboard();
    toggleAutoRefresh();
  }

  function switchToChat() {
    if (_autoRefreshTimer) { clearInterval(_autoRefreshTimer); _autoRefreshTimer = null; }
    state.adminMode = false;
    $('#sidebar-admin').classList.add('hidden');
    $('#sidebar-chat').classList.remove('hidden');
    $('#admin-view').classList.add('hidden');
    $('#chat-view').classList.remove('hidden');
  }

  function getAdminHours() {
    const sel = $('#admin-time-range');
    return sel ? parseInt(sel.value, 10) : 168;
  }

  function showAdminTab(tabName) {
    state.adminTab = tabName;
    document.querySelectorAll('.admin-tab').forEach(t =>
      t.classList.toggle('active', t.dataset.tab === tabName));
    const dashPanel = $('#panel-dashboard');
    const reqPanel = $('#panel-requests');
    if (dashPanel) dashPanel.classList.toggle('hidden', tabName !== 'dashboard');
    if (reqPanel) reqPanel.classList.toggle('hidden', tabName !== 'requests');
    if (tabName === 'requests') loadRequestsPanel();
  }

  function loadRequestsPanel() {
    if (state.requestMode === 'low') {
      loadLowScores();
    } else {
      loadAllRequests();
    }
  }

  async function loadAllRequests() {
    const offset = state.lowScorePage * state.lowScorePageSize;
    const limit = state.lowScorePageSize;
    try {
      const json = await apiFetch(API + '/eval/dashboard/requests?limit=' + limit + '&offset=' + offset);
      const data = json.data || [];
      state.lowScoreData = data;
      populateTypeFilter(data);
      applyLowScoreFilters();
      updateLowScorePagination(data.length);
    } catch {
      updateLowScorePagination(0);
    }
  }

  let _autoRefreshTimer = null;

  function nextAnimationFrame() {
    return new Promise((resolve) => window.requestAnimationFrame(resolve));
  }

  function toggleAutoRefresh() {
    if (_autoRefreshTimer) {
      clearInterval(_autoRefreshTimer);
      _autoRefreshTimer = null;
    }
    const checked = $('#admin-auto-refresh').checked;
    if (checked && state.adminMode) {
      _autoRefreshTimer = setInterval(() => {
        if (state.adminMode) {
          loadAdminDashboard();
          if (state.adminTab === 'requests') loadRequestsPanel();
        }
      }, 60000);
    }
  }

  function exportDashboardCsv() {
    const hours = getAdminHours();
    const userId = ($('#admin-user-filter')?.value || '').trim();
    const fmt = $('#admin-export-format')?.value || 'csv';
    let url = API + '/eval/dashboard/export?hours=' + hours + '&format=' + fmt;
    if (userId) url += '&user_id=' + encodeURIComponent(userId);
    window.open(url, '_blank');
  }

  async function loadAdminDashboard() {
    const loading = $('#admin-loading');
    const stats = $('#admin-stats');
    loading.classList.remove('hidden');
    stats.classList.add('hidden');

    // Show skeletons
    loading.innerHTML = '<div class="stats-grid stats-grid-4">' +
      '<div class="skeleton skeleton-card"></div>'.repeat(8) + '</div>' +
      '<div class="chart-row"><div class="skeleton skeleton-chart" style="flex:3"></div><div class="skeleton skeleton-chart" style="flex:2"></div></div>';

    try {
      const hours = getAdminHours();
      const granularity = hours <= 24 ? 'hourly' : 'daily';

      const userId = ($('#admin-user-filter')?.value || '').trim() || '';
      const userParam = userId ? '&user_id=' + encodeURIComponent(userId) : '';
      const [onlineResp, trendResp] = await Promise.all([
        apiFetch(API + '/eval/dashboard/online?hours=' + hours + userParam),
        apiFetch(API + '/eval/dashboard/score-trend?hours=' + hours + '&granularity=' + granularity + userParam),
      ]);

      state._lastOnlineResp = onlineResp;
      state._lastTrendResp = trendResp;
      loading.classList.add('hidden');
      stats.classList.remove('hidden');
      await nextAnimationFrame();

      renderAdminStats(onlineResp);
      renderTrendLegend();
      renderScoreTrend(trendResp);
      renderByReportType(onlineResp);
      renderMemoryStats(onlineResp);

      // Populate user datalist
      try {
        const usersResp = await apiFetch(API + '/eval/dashboard/users');
        const datalist = $('#user-list');
        if (datalist && usersResp.data) {
          datalist.innerHTML = usersResp.data.map((u) => '<option value="' + escapeHtml(u) + '">').join('');
        }
      } catch {}
    } catch (err) {
      loading.textContent = 'Failed to load dashboard: ' + err.message;
    }
  }

  // --- ① Stat Cards ---
  function renderAdminStats(resp) {
    const d = resp.data || {};
    const total = d.total_requests || 0;

    $('#stat-total').textContent = total;
    setScoreCard('stat-avg-score', d.avg_overall_score);
    setScoreCard('stat-memory-score', d.avg_memory_score);
    setScoreCard('stat-answer-quality', d.avg_answer_quality_score);

    const avgLatencyEl = document.getElementById('stat-avg-latency');
    if (avgLatencyEl) { avgLatencyEl.textContent = (d.avg_latency_ms || 0) + ' ms'; avgLatencyEl.classList.remove('score-low', 'score-medium', 'score-high'); }

    const avgTokenEl = document.getElementById('stat-avg-token');
    if (avgTokenEl) { avgTokenEl.textContent = (d.avg_total_tokens || 0).toLocaleString(); avgTokenEl.classList.remove('score-low', 'score-medium', 'score-high'); }

    const successEl = document.getElementById('stat-success-rate');
    if (successEl) {
      const rate = total > 0 ? ((total - (d.has_error_count || 0)) / total * 100) : 0;
      successEl.textContent = rate.toFixed(1) + '%';
      successEl.classList.remove('score-low', 'score-medium', 'score-high');
    }
  }

  function setBoolCountCard(id, count) {
    const el = document.getElementById(id);
    if (!el) return;
    const c = count || 0;
    el.textContent = c;
    el.classList.remove('score-low', 'score-medium', 'score-high');
    el.classList.add(c > 0 ? 'score-medium' : 'score-high');
  }

  async function loadLowScores() {
    const offset = state.lowScorePage * state.lowScorePageSize;
    const limit = state.lowScorePageSize;
    try {
      const json = await apiFetch(API + '/eval/dashboard/low-scores?threshold=0.4&limit=' + limit + '&offset=' + offset);
      const data = json.data || [];
      state.lowScoreData = data;
      populateTypeFilter(data);
      applyLowScoreFilters();
      updateLowScorePagination(data.length);
    } catch {
      updateLowScorePagination(0);
    }
  }

  function populateTypeFilter(data) {
    const sel = $('#ls-type-filter');
    if (!sel) return;
    const types = new Set(data.map((d) => d.query_type).filter(Boolean));
    const current = sel.value;
    // Keep "All Types" option, rebuild the rest
    sel.innerHTML = '<option value="">All Types</option>';
    [...types].sort().forEach((t) => {
      sel.innerHTML += '<option value="' + escapeHtml(t) + '">' + escapeHtml(t) + '</option>';
    });
    sel.value = current;
  }

  function applyLowScoreFilters() {
    let data = state.lowScoreData.slice();
    // Search filter
    if (state.lowScoreSearch) {
      data = data.filter((d) => (d.query || '').toLowerCase().includes(state.lowScoreSearch));
    }
    // Type filter
    if (state.lowScoreTypeFilter) {
      data = data.filter((d) => d.query_type === state.lowScoreTypeFilter);
    }
    // Sort
    const key = state.lowScoreSortKey;
    const dir = state.lowScoreSortDir === 'asc' ? 1 : -1;
    data.sort((a, b) => {
      let va = a[key], vb = b[key];
      if (key === 'created_at') {
        va = va ? new Date(va).getTime() : 0;
        vb = vb ? new Date(vb).getTime() : 0;
      } else {
        va = va ?? 0;
        vb = vb ?? 0;
      }
      if (va < vb) return -1 * dir;
      if (va > vb) return 1 * dir;
      return 0;
    });
    // Update sort indicators
    $$('.low-scores-section th.sortable').forEach((th) => {
      th.classList.remove('asc', 'desc');
      const arrow = th.querySelector('.sort-arrow');
      if (arrow) arrow.textContent = '\u25B2';
      if (th.dataset.sort === key) {
        th.classList.add(state.lowScoreSortDir);
        if (arrow) arrow.textContent = '\u25B2';
      }
    });
    renderLowScores({ data });
  }

  function updateLowScorePagination(itemsReturned) {
    const pagination = $('#low-scores-pagination');
    if (!pagination) return;
    pagination.classList.remove('hidden');
    const prevBtn = $('#ls-prev');
    const nextBtn = $('#ls-next');
    const info = $('#ls-page-info');
    prevBtn.disabled = state.lowScorePage === 0;
    nextBtn.disabled = itemsReturned < state.lowScorePageSize;
    info.textContent = 'Page ' + (state.lowScorePage + 1);
  }

  function setScoreCard(id, value) {
    const el = document.getElementById(id);
    if (!el) return;
    if (value !== null && value !== undefined) {
      el.textContent = value.toFixed(2);
      el.classList.remove('score-low', 'score-medium', 'score-high');
      el.classList.add(value < 0.4 ? 'score-low' : value < 0.6 ? 'score-medium' : 'score-high');
    } else {
      el.textContent = '--';
      el.classList.remove('score-low', 'score-medium', 'score-high');
    }
  }

  // --- ② Multi-line Trend ---
  function renderTrendLegend() {
    const legend = $('#trend-legend');
    if (!legend) return;
    legend.innerHTML = '';
    TREND_LINES.forEach((line) => {
      const item = document.createElement('span');
      const color = trendLineColor(line);
      item.className = 'legend-item' + (line.active ? ' active' : '');
      item.innerHTML = `<span class="legend-dot" style="background:${color}"></span>${escapeHtml(line.label)}`;
      item.addEventListener('click', () => {
        line.active = !line.active;
        item.classList.toggle('active', line.active);
        // Re-render trend from cached data
        if (state._lastTrendResp) renderScoreTrend(state._lastTrendResp);
      });
      legend.appendChild(item);
    });
  }

  function getChartWidth(canvas, minWidth) {
    const parent = canvas?.parentElement;
    if (!parent) return minWidth;
    const styles = window.getComputedStyle(parent);
    const paddingX = (parseFloat(styles.paddingLeft) || 0) + (parseFloat(styles.paddingRight) || 0);
    return Math.max(minWidth, parent.clientWidth - paddingX);
  }

  function clampValue(value, min, max) {
    return Math.min(Math.max(value, min), max);
  }

  function formatAdminLabel(label) {
    return String(label || '').replace(/_/g, ' ');
  }

  function truncateCanvasText(ctx, text, maxWidth) {
    const source = String(text || '');
    if (!source || ctx.measureText(source).width <= maxWidth) return source;
    const ellipsis = '...';
    let end = source.length;
    while (end > 0 && ctx.measureText(source.slice(0, end) + ellipsis).width > maxWidth) {
      end -= 1;
    }
    return end > 0 ? source.slice(0, end) + ellipsis : ellipsis;
  }

  function formatTrendLabel(label) {
    const value = String(label || '');
    if (/^\d{4}-\d{2}-\d{2}$/.test(value)) return value.slice(5);
    if (/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$/.test(value)) return value.slice(5).replace('T', ' ');
    return value;
  }

  function scoreColor(value) {
    const light = document.documentElement.getAttribute('data-theme') === 'light';
    if (value < 0.4) return light ? '#e8384f' : '#ff4757';
    if (value < 0.6) return light ? '#d48a20' : '#f0a030';
    return light ? '#00b882' : '#00e5a0';
  }

  function canvasColors() {
    const light = document.documentElement.getAttribute('data-theme') === 'light';
    return light ? {
      muted: '#8896a8', secondary: '#4a5568', primary: '#1a1f36',
      grid: 'rgba(0,0,0,0.06)', track: 'rgba(0,0,0,0.04)',
    } : {
      muted: '#4d6580', secondary: '#8fa4be', primary: '#e0e8f0',
      grid: 'rgba(255,255,255,0.06)', track: 'rgba(255,255,255,0.04)',
    };
  }

  function renderScoreTrend(resp) {
    state._lastTrendResp = resp;
    const canvas = $('#score-trend-chart');
    const ctx = canvas.getContext('2d');
    const data = resp.data || [];
    const tooltip = $('#chart-tooltip');

    const dpr = window.devicePixelRatio || 1;
    const cw = getChartWidth(canvas, 320);
    const ch = 240;
    canvas.width = cw * dpr;
    canvas.height = ch * dpr;
    canvas.style.width = cw + 'px';
    canvas.style.height = ch + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const w = cw;
    const h = ch;
    const pad = { top: 16, right: 16, bottom: 36, left: 44 };
    const plotW = w - pad.left - pad.right;
    const plotH = h - pad.top - pad.bottom;

    ctx.clearRect(0, 0, w, h);

    const cc = canvasColors();

    if (!data.length) {
      ctx.font = '13px "DM Sans", sans-serif';
      ctx.fillStyle = cc.muted;
      ctx.textAlign = 'center';
      ctx.fillText('No trend data available', w / 2, h / 2);
      return;
    }

    const labels = data.map((d) => d.period ?? d.time ?? d.date ?? '');
    const displayLabels = labels.map(formatTrendLabel);
    const allPoints = [];
    const xForIndex = (index, total) => {
      if (total <= 1) return pad.left + plotW / 2;
      return pad.left + plotW * index / (total - 1);
    };

    // Grid
    ctx.strokeStyle = cc.grid;
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = pad.top + plotH * i / 4;
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(w - pad.right, y);
      ctx.stroke();
      ctx.fillStyle = cc.muted;
      ctx.font = '10px "DM Sans", sans-serif';
      ctx.textAlign = 'right';
      ctx.fillText((1 - i * 0.25).toFixed(1), pad.left - 6, y + 3);
    }

    // X labels
    ctx.textAlign = 'center';
    ctx.fillStyle = cc.muted;
    ctx.font = '10px "DM Sans", sans-serif';
    const maxLabelCount = Math.max(1, Math.floor(plotW / 70));
    const step = Math.max(1, Math.ceil(labels.length / maxLabelCount));
    displayLabels.forEach((label, i) => {
      if (i % step === 0) {
        const x = xForIndex(i, displayLabels.length);
        ctx.fillText(label, x, h - 6);
      }
    });

    // Draw active lines and collect points
    TREND_LINES.forEach((line) => {
      if (!line.active) return;
      const color = trendLineColor(line);
      const points = data.map((d) => d[line.key] ?? 0);

      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      points.forEach((p, i) => {
        const x = xForIndex(i, points.length);
        const y = pad.top + plotH * (1 - Math.min(Math.max(p, 0), 1));
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
        allPoints.push({ x, y, idx: i, label: labels[i], value: p, color: color, lineLabel: line.label });
      });
      if (points.length > 1) ctx.stroke();

      // Dots
      points.forEach((p, i) => {
        const x = xForIndex(i, points.length);
        const y = pad.top + plotH * (1 - Math.min(Math.max(p, 0), 1));
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
      });
    });

    // Tooltip mousemove handler (bind once)
    if (!canvas._tooltipBound) {
      canvas._tooltipBound = true;
      canvas.addEventListener('mousemove', (e) => {
        const cr = canvas.getBoundingClientRect();
        const mx = e.clientX - cr.left;
        const my = e.clientY - cr.top;
        const nearest = findNearestPoints(canvas._allPoints || [], mx, my, 20);
        if (nearest.length) {
          const period = nearest[0].label;
          let html = '<div class="tt-period">' + escapeHtml(period) + '</div>';
          nearest.forEach((p) => {
            html += '<div class="tt-line"><span class="tt-dot" style="background:' + p.color + '"></span>' + escapeHtml(p.lineLabel) + ': ' + p.value.toFixed(3) + '</div>';
          });
          tooltip.innerHTML = html;
          tooltip.style.display = 'block';
          const sectionRect = canvas.closest('.chart-section').getBoundingClientRect();
          let tx = e.clientX - sectionRect.left + 12;
          let ty = e.clientY - sectionRect.top - 10;
          if (tx + 160 > sectionRect.width) tx = tx - 170;
          tooltip.style.left = tx + 'px';
          tooltip.style.top = ty + 'px';
        } else {
          tooltip.style.display = 'none';
        }
      });
      canvas.addEventListener('mouseleave', () => {
        tooltip.style.display = 'none';
      });
    }
    canvas._allPoints = allPoints;
  }

  function findNearestPoints(points, mx, my, threshold) {
    const byIdx = {};
    points.forEach((p) => {
      const dist = Math.sqrt((p.x - mx) ** 2 + (p.y - my) ** 2);
      if (dist < threshold) {
        if (!byIdx[p.idx] || byIdx[p.idx]._dist > dist) {
          byIdx[p.idx] = { ...p, _dist: dist };
        }
      }
    });
    return Object.values(byIdx).sort((a, b) => a._dist - b._dist);
  }

  // --- ② Request Type Distribution ---
  function renderByReportType(resp) {
    const canvas = $('#query-type-chart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const byType = (resp.data || {}).by_query_type || {};
    const entries = Object.entries(byType).sort((a, b) => b[1].count - a[1].count);

    const dpr = window.devicePixelRatio || 1;
    const cw = getChartWidth(canvas, 280);
    const barH = 28;
    const gap = 10;
    const ch = Math.max(140, entries.length * (barH + gap) + 20);
    canvas.width = cw * dpr;
    canvas.height = ch * dpr;
    canvas.style.width = cw + 'px';
    canvas.style.height = ch + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    ctx.clearRect(0, 0, cw, ch);

    const cc = canvasColors();

    if (!entries.length) {
      ctx.font = '13px "DM Sans", sans-serif';
      ctx.fillStyle = cc.muted;
      ctx.textAlign = 'center';
      ctx.fillText('No data', cw / 2, ch / 2);
      return;
    }

    const totalCount = entries.reduce((s, [, v]) => s + v.count, 0);
    const metaTexts = entries.map(([, info]) => {
      const pct = totalCount > 0 ? info.count / totalCount : 0;
      const score = info.avg_overall_score !== undefined ? info.avg_overall_score.toFixed(2) : '--';
      return info.count + ' • ' + (pct * 100).toFixed(0) + '% • ' + score;
    });
    ctx.font = '12px "DM Sans", sans-serif';
    const labelW = clampValue(cw * 0.28, 80, 140);
    const metaW = clampValue(
      Math.max(...metaTexts.map((text) => ctx.measureText(text).width)) + 8,
      90,
      140
    );
    const barX = labelW + 16;
    const barMaxW = Math.max(48, cw - barX - metaW - 12);

    entries.forEach(([type, info], i) => {
      const y = i * (barH + gap) + 8;
      const cy = y + barH / 2;
      const pct = totalCount > 0 ? info.count / totalCount : 0;
      const bw = barMaxW * pct;
      const label = truncateCanvasText(ctx, formatAdminLabel(type), labelW);

      // Label
      ctx.fillStyle = cc.secondary;
      ctx.font = '12px "DM Sans", sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText(label, 0, cy + 4);

      // Track
      ctx.fillStyle = cc.track;
      ctx.beginPath();
      ctx.roundRect(barX, y + 4, barMaxW, barH - 8, 6);
      ctx.fill();

      // Bar
      ctx.fillStyle = scoreColor(info.avg_overall_score ?? 0);
      ctx.beginPath();
      ctx.roundRect(barX, y + 4, bw, barH - 8, 6);
      ctx.fill();

      // Count + score
      ctx.fillStyle = cc.primary;
      ctx.font = '11px "DM Sans", sans-serif';
      ctx.textAlign = 'right';
      ctx.fillText(metaTexts[i], cw, cy + 4);
    });
  }

  // --- ③ Memory Metrics ---
  function renderMemoryStats(resp) {
    const container = $('#memory-stats');
    if (!container) return;
    const d = (resp && resp.data) || {};

    const metrics = [
      { label: 'Average Memory Score', value: d.avg_memory_score, fmt: (v) => Number(v).toFixed(3) },
      { label: 'Recorded Memories', value: d.recorded_memory_count, fmt: (v) => Number(v || 0).toLocaleString() },
      { label: 'Duplicate Ratio', value: d.memory_duplicate_ratio, fmt: (v) => (Number(v) * 100).toFixed(1) + '%' },
      { label: 'Conflict Ratio', value: d.memory_conflict_ratio, fmt: (v) => (Number(v) * 100).toFixed(1) + '%' },
    ];

    container.innerHTML = metrics.map((m) => {
      const display = (m.value !== null && m.value !== undefined)
        ? m.fmt(m.value) : '--';
      return `<div class="memory-item">
        <div class="memory-item-label">${escapeHtml(m.label)}</div>
        <div class="memory-item-value">${display}</div>
      </div>`;
    }).join('');
  }

  // --- ④ Low Score Table (enhanced) ---
  function renderLowScores(resp) {
    const tbody = $('#low-scores-body');
    const data = resp.data || [];
    if (!data.length) {
      tbody.innerHTML = '<tr><td colspan="8" style="text-align:center;color:#4d6580">No low score alerts</td></tr>';
      return;
    }
    tbody.innerHTML = '';
    data.forEach((d) => {
      const overall = d.overall_score;
      const memory = d.memory_score;
      const answer = d.answer_quality_score;

      // Summary row
      const row = document.createElement('tr');
      row.className = 'low-score-row';
      row.innerHTML = `
        <td><span class="expand-arrow">&#9654;</span></td>
        <td class="query-text" title="${escapeHtml(d.query || '')}">${escapeHtml((d.query || '').substring(0, 60))}</td>
        <td>${escapeHtml(d.query_type || '--')}</td>
        <td>${scoreBadge(overall)}</td>
        <td>${scoreBadge(memory)}</td>
        <td>${scoreBadge(answer)}</td>
        <td>${escapeHtml(formatTime(d.created_at))}</td>
      `;

      // Expand row
      const expandRow = document.createElement('tr');
      expandRow.className = 'low-score-expand';
      expandRow.style.display = 'none';
      const expandCell = document.createElement('td');
      expandCell.colSpan = 7;
      expandCell.innerHTML = `<div class="expand-content" id="detail-${escapeHtml(d.request_id)}"><div class="dim-loading">Loading...</div></div>`;
      expandRow.appendChild(expandCell);

      row.addEventListener('click', () => {
        const open = expandRow.style.display !== 'none';
        expandRow.style.display = open ? 'none' : '';
        row.querySelector('.expand-arrow').classList.toggle('open', !open);
        if (!open) fetchRequestDetail(d.request_id);
      });

      tbody.appendChild(row);
      tbody.appendChild(expandRow);
    });
  }

  function scoreBadge(val) {
    if (val === null || val === undefined) {
      return '<span class="score-badge">--</span>';
    }
    const v = typeof val === 'number' ? val : 0;
    const cls = v < 0.3 ? 'low' : v < 0.6 ? 'medium' : 'high';
    return `<span class="score-badge ${cls}">${v.toFixed(2)}</span>`;
  }

  function formatTime(iso) {
    if (!iso) return '--';
    try {
      const d = new Date(iso);
      return d.toLocaleDateString() + ' ' + d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } catch {
      return iso;
    }
  }

  async function fetchRequestDetail(requestId) {
    const container = document.getElementById('detail-' + requestId);
    if (!container || container.dataset.loaded) return;
    try {
      const [evalResp, eventsResp, diagResp, memoryResp] = await Promise.all([
        apiFetch(`${API}/eval/requests/${encodeURIComponent(requestId)}`),
        apiFetch(`${API}/eval/requests/${encodeURIComponent(requestId)}/events`),
        apiFetchOptional(`${API}/eval/requests/${encodeURIComponent(requestId)}/diagnosis`),
        apiFetchOptional(`${API}/eval/memory/requests/${encodeURIComponent(requestId)}`),
      ]);
      renderDetailSections(
        container,
        evalResp.data || evalResp,
        eventsResp.data || eventsResp,
        diagResp ? (diagResp.data || diagResp) : null,
        memoryResp ? (memoryResp.data || memoryResp) : null,
      );
      container.dataset.loaded = '1';
    } catch {
      container.innerHTML = '<div class="dim-loading">Failed to load detail data.</div>';
    }
  }

  function renderDetailSections(container, evalData, eventsData, diagData, memoryData) {
    const eh = evalData.execution_health || {};
    const rt = evalData.runtime_metrics || {};
    const memRun = (memoryData && memoryData.run) || {};
    const retrievalEvent = (memoryData && memoryData.retrieval_event) || {};
    const extractionEvent = (memoryData && memoryData.extraction_event) || {};
    const hasMemoryV2Data = Boolean(memoryData && (memoryData.run || memoryData.retrieval_event || memoryData.extraction_event));

    let html = '';

    // --- Dimension nav ---
    html += '<div class="dim-nav">';
    html += '<span class="dim-nav-item dim-nav-req" data-dim="dim-sec-req">Request</span>';
    if (hasMemoryV2Data) {
      html += '<span class="dim-nav-item dim-nav-mem" data-dim="dim-sec-mem">Memory</span>';
    }
    html += '<span class="dim-nav-item dim-nav-eval" data-dim="dim-sec-eval">Evaluation</span>';
    html += '</div>';

    // --- Dimension 1: Request Overview ---
    const healthScore = evalData.execution_health_score ?? eh.execution_health_score;
    const reqMetrics = [
      { label: 'Total Latency', value: eh.total_latency_ms, fmt: 'ms' },
      { label: 'Health Score', value: healthScore, fmt: 'score' },
      { label: 'Prompt Tokens', value: rt.prompt_tokens, fmt: 'tokens' },
      { label: 'Completion Tokens', value: rt.completion_tokens, fmt: 'tokens' },
      { label: 'Total Tokens', value: rt.total_tokens, fmt: 'tokens' },
      { label: 'Overall Score', value: evalData.overall_score, fmt: 'score' },
      { label: 'Answer Quality', value: evalData.answer_quality_score, fmt: 'score' },
      { label: 'Has Error', value: eh.has_error, fmt: 'bool' },
      { label: 'Has Retry', value: eh.has_retry, fmt: 'bool' },
      { label: 'Has Fallback', value: eh.has_fallback, fmt: 'bool' },
      { label: 'Timeout', value: eh.timeout, fmt: 'bool' },
    ];
    html += '<div class="dim-section" id="dim-sec-req">'
      + '<div class="dim-header dim-req" data-dim-toggle="dim-sec-req"><span class="dim-toggle">&#9660;</span>Request Overview</div>'
      + '<div class="dim-body">' + renderDimGrid(reqMetrics) + '</div>'
      + '</div>';

    // --- Dimension 2: Memory Data ---
    if (hasMemoryV2Data) {
      const memMetrics = [
        { label: 'Memory Score', value: memRun.memory_score, fmt: 'score' },
        { label: 'Memory Hit@K', value: memRun.memory_hit_at_k, fmt: 'score' },
        { label: 'Relevant Recall', value: memRun.memory_relevant_recall, fmt: 'score' },
        { label: 'Relevant Precision', value: memRun.memory_relevant_precision, fmt: 'score' },
        { label: 'First Relevant Rank', value: memRun.first_relevant_rank, fmt: 'count' },
        { label: 'Stale Retrieval Rate', value: memRun.memory_stale_retrieval_rate, fmt: 'pct' },
        { label: 'Answer Relevance', value: memRun.memory_answer_relevance, fmt: 'score' },
        { label: 'Extraction Precision', value: memRun.memory_extraction_precision, fmt: 'score' },
        { label: 'Injected Count', value: memRun.memory_injected_count ?? retrievalEvent.injected_count, fmt: 'count' },
        { label: 'Injected Latency', value: memRun.memory_injected_latency_ms ?? retrievalEvent.vector_search_latency_ms, fmt: 'ms' },
        { label: 'Injected Tokens', value: memRun.memory_injected_tokens ?? retrievalEvent.injected_tokens, fmt: 'tokens' },
        { label: 'Extraction Latency', value: memRun.memory_extraction_latency_ms ?? extractionEvent.dispatch_latency_ms, fmt: 'ms' },
        { label: 'Extraction Tokens', value: memRun.memory_extraction_tokens ?? extractionEvent.total_tokens, fmt: 'tokens' },
      ];
      html += '<div class="dim-section" id="dim-sec-mem">'
        + '<div class="dim-header dim-mem" data-dim-toggle="dim-sec-mem"><span class="dim-toggle">&#9660;</span>Memory Data</div>'
        + '<div class="dim-body">' + renderDimGrid(memMetrics);
      if (retrievalEvent && retrievalEvent.event_id) {
        html += renderDimEventsTable([{
          retrieved_count: retrievalEvent.retrieved_count,
          embedding_latency_ms: retrievalEvent.embedding_latency_ms,
          injected_count: retrievalEvent.injected_count,
          vector_search_latency_ms: retrievalEvent.vector_search_latency_ms,
          injected_tokens: retrievalEvent.injected_tokens,
          retrieved_memory_ids: (retrievalEvent.retrieved_memory_ids || []).join(', '),
          injected_memory_ids: (retrievalEvent.injected_memory_ids || []).join(', '),
        }], [
          { key: 'retrieved_count', label: 'Retrieved Count', fmt: 'count' },
          { key: 'embedding_latency_ms', label: 'Embedding Latency', fmt: 'ms' },
          { key: 'injected_count', label: 'Injected', fmt: 'count' },
          { key: 'vector_search_latency_ms', label: 'Vector Search', fmt: 'ms' },
          { key: 'injected_tokens', label: 'Injected Tokens', fmt: 'tokens' },
          { key: 'retrieved_memory_ids', label: 'Retrieved IDs' },
          { key: 'injected_memory_ids', label: 'Injected IDs' },
        ]);
      }
      if (retrievalEvent && retrievalEvent.injected_text) {
        html += renderDimEventsTable([{
          injected_text: retrievalEvent.injected_text,
        }], [
          { key: 'injected_text', label: 'Injected Text', render: renderInjectedText },
        ]);
      }
      html += '</div></div>';
    }

    // --- Dimension 3: Evaluation ---
    const _diag = diagData || {};
    const issues = (_diag.issues || []);
    const strengths = (_diag.strengths || []);
    const recs = (_diag.recommendations || []);
    html += '<div class="dim-section" id="dim-sec-eval">'
      + '<div class="dim-header dim-eval" data-dim-toggle="dim-sec-eval"><span class="dim-toggle">&#9660;</span>Evaluation</div>'
      + '<div class="dim-body">';
    if (issues.length || strengths.length || recs.length) {
      const light = document.documentElement.getAttribute('data-theme') === 'light';
      const cIssue = light ? '#e8384f' : '#ff4757';
      const cStrength = light ? '#00b882' : '#00e5a0';
      const cRec = light ? '#3b6cf5' : '#4f7cff';
      if (issues.length) {
        html += '<div style="margin-bottom:6px;font-size:12px;color:' + cIssue + ';font-weight:600">Issues</div>'
          + '<ul class="dim-list">' + issues.map((s) => '<li class="dim-issue">' + escapeHtml(s) + '</li>').join('') + '</ul>';
      }
      if (strengths.length) {
        html += '<div style="margin-bottom:6px;font-size:12px;color:' + cStrength + ';font-weight:600">Strengths</div>'
          + '<ul class="dim-list">' + strengths.map((s) => '<li class="dim-strength">' + escapeHtml(s) + '</li>').join('') + '</ul>';
      }
      if (recs.length) {
        html += '<div style="margin-bottom:6px;font-size:12px;color:' + cRec + ';font-weight:600">Recommendations</div>'
          + '<ul class="dim-list">' + recs.map((s) => '<li class="dim-rec">' + escapeHtml(s) + '</li>').join('') + '</ul>';
      }
    } else {
      html += '<div style="font-size:12px;color:var(--text-muted)">No evaluation data available.</div>';
    }
    html += '</div></div>';

    container.innerHTML = html;

    // Bind dimension header collapse toggles
    container.querySelectorAll('.dim-header[data-dim-toggle]').forEach((header) => {
      header.addEventListener('click', (e) => {
        // Don't collapse if clicking on the nav item
        if (e.target.closest('.dim-nav-item')) return;
        const sectionId = header.dataset.dimToggle;
        const section = container.querySelector('#' + sectionId);
        if (section) {
          section.classList.toggle('collapsed');
          header.classList.toggle('collapsed');
        }
      });
    });

    // Bind nav items to scroll-to and highlight
    container.querySelectorAll('.dim-nav-item').forEach((navItem) => {
      navItem.addEventListener('click', (e) => {
        e.stopPropagation();
        const dimId = navItem.dataset.dim;
        const section = container.querySelector('#' + dimId);
        if (section) {
          // Ensure section is expanded
          section.classList.remove('collapsed');
          const header = section.querySelector('.dim-header');
          if (header) header.classList.remove('collapsed');
          // Scroll into view within the expand-content container
          section.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      });
    });
  }

  function renderDimGrid(metrics) {
    const cells = metrics.map((m) => {
      const display = formatMetric(m.value, m.fmt);
      return `<div class="dim-item"><div class="dim-item-label">${escapeHtml(m.label)}</div><div class="dim-item-value">${display}</div></div>`;
    });
    return '<div class="dim-grid">' + cells.join('') + '</div>';
  }

  function renderDimEventsTable(events, columns) {
    if (!events.length) return '';
    const ths = columns.map((c) => '<th>' + escapeHtml(c.label) + '</th>').join('');
    const rows = events.map((ev) => {
      const tds = columns.map((c) => {
        const raw = ev[c.key];
        const val = raw !== undefined && raw !== null ? raw : '';
        if (c.key === 'returned_paper_ids' && Array.isArray(val)) {
          return '<td>' + val.map((p) => escapeHtml(String(p))).join(', ') + '</td>';
        }
        if (typeof c.render === 'function') {
          return '<td>' + c.render(val) + '</td>';
        }
        return '<td>' + (c.fmt ? formatMetric(val, c.fmt) : escapeHtml(String(val))) + '</td>';
      }).join('');
      return '<tr>' + tds + '</tr>';
    }).join('');
    return '<table class="dim-events-table"><thead><tr>' + ths + '</tr></thead><tbody>' + rows + '</tbody></table>';
  }

  function renderInjectedText(value) {
    const lines = String(value)
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean);
    return lines.map((line) => '<div class="dim-memory-line">' + escapeHtml(line) + '</div>').join('');
  }

  function formatMetric(value, type) {
    if (value === null || value === undefined) return '--';
    switch (type) {
      case 'ms':    return (typeof value === 'number' ? value.toLocaleString() + 'ms' : escapeHtml(String(value)));
      case 'pct':   return (typeof value === 'number' ? (value * 100).toFixed(1) + '%' : escapeHtml(String(value)));
      case 'score': return (typeof value === 'number' ? value.toFixed(3) : escapeHtml(String(value)));
      case 'bool':  return boolBadge(value);
      case 'tokens':
      case 'count': return (typeof value === 'number' ? value.toLocaleString() : escapeHtml(String(value)));
      case 'text':  return '<span class="dim-tag dim-tag-strategy">' + escapeHtml(String(value)) + '</span>';
      default:      return escapeHtml(String(value));
    }
  }

  function boolBadge(val) {
    if (val === null || val === undefined) return '<span style="color:var(--text-muted)">--</span>';
    if (val) return '<span class="dim-tag dim-tag-bool-true">Yes</span>';
    return '<span class="dim-tag dim-tag-bool-false">No</span>';
  }

  // ----- Utilities -----
  function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  function renderMarkdown(text) {
    if (!text) return '';
    try {
      if (typeof marked !== 'undefined') {
        const html = marked.parse(text);
        return typeof DOMPurify !== 'undefined' ? DOMPurify.sanitize(html) : html;
      }
    } catch { /* fallback */ }
    return text.split('\n').map((l) => `<p>${escapeHtml(l)}</p>`).join('');
  }

  // Listen for input changes to update send button
  document.addEventListener('input', (e) => {
    if (e.target.id === 'message-input') updateSendButton();
  });
})();
