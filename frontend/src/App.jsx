import React, { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import ChatInterface from './components/ChatInterface';
import Login from './components/Login';

import {
  getCurrentUser,
  fetchAuthSession,
  fetchUserAttributes,
  signOut as cognitoSignOut
} from 'aws-amplify/auth';
import { Hub } from 'aws-amplify/utils';

import './index.css';

function App() {
  /* ──────────────────────────────────────────────────────────────
   * Theme
   * ────────────────────────────────────────────────────────────── */
  const [theme, setTheme] = useState(() => {
    try {
      return localStorage.getItem('akasha_theme') || 'dark';
    } catch (_) {
      return 'dark';
    }
  });

  useEffect(() => {
    document.documentElement.setAttribute(
      'data-theme',
      theme
    );

    try {
      localStorage.setItem('akasha_theme', theme);
    } catch (_) {}
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prev =>
      prev === 'dark' ? 'light' : 'dark'
    );
  };

  /* ──────────────────────────────────────────────────────────────
   * Authentication
   * ────────────────────────────────────────────────────────────── */
  const [user, setUser] = useState(null);
  const [authLoading, setAuthLoading] = useState(true);

  /* ──────────────────────────────────────────────────────────────
   * Chat sessions
   * ────────────────────────────────────────────────────────────── */
  const [sessions, setSessions] = useState([]);
  const [activeSessionId, setActiveSessionId] =
    useState(null);

  /* ──────────────────────────────────────────────────────────────
   * Per-session draft cache
   * ────────────────────────────────────────────────────────────── */
  const [drafts, setDrafts] = useState(() => {
    try {
      const saved =
        localStorage.getItem('akasha_drafts');

      if (saved) {
        const parsed = JSON.parse(saved);
        const restored = {};

        Object.entries(parsed).forEach(
          ([sid, draft]) => {
            restored[sid] = {
              query: draft.query || '',
              files: []
            };
          }
        );

        return restored;
      }
    } catch (_) {}

    return {};
  });

  const currentDraft =
    drafts[activeSessionId] || {
      query: '',
      files: []
    };

  const selectedFiles = currentDraft.files;

  const setSelectedFiles = files => {
    if (!activeSessionId) return;

    setDrafts(prev => ({
      ...prev,
      [activeSessionId]: {
        ...(prev[activeSessionId] || {
          query: '',
          files: []
        }),
        files
      }
    }));
  };

  const setDraftQuery = query => {
    if (!activeSessionId) return;

    setDrafts(prev => ({
      ...prev,
      [activeSessionId]: {
        ...(prev[activeSessionId] || {
          query: '',
          files: []
        }),
        query
      }
    }));
  };

  const clearDraft = () => {
    if (!activeSessionId) return;

    setDrafts(prev => ({
      ...prev,
      [activeSessionId]: {
        query: '',
        files: []
      }
    }));
  };

  /* ──────────────────────────────────────────────────────────────
   * Persist drafts
   * ────────────────────────────────────────────────────────────── */
  useEffect(() => {
    try {
      const toSave = {};

      Object.entries(drafts).forEach(
        ([sid, draft]) => {
          if (draft.query) {
            toSave[sid] = {
              query: draft.query
            };
          }
        }
      );

      localStorage.setItem(
        'akasha_drafts',
        JSON.stringify(toSave)
      );
    } catch (_) {}
  }, [drafts]);

  /* ──────────────────────────────────────────────────────────────
   * Authentication initialization
   *
   * Cognito:
   *   - email/password users
   *   - Google federated users
   *
   * Demo login is handled directly by Login.jsx.
   * ────────────────────────────────────────────────────────────── */
  useEffect(() => {
    let mounted = true;

    const initializeAuth = async () => {
      try {
        const currentUser = await getCurrentUser();
        const session = await fetchAuthSession();

        if (!mounted) return;

        const accessToken =
          session.tokens?.accessToken?.toString() ||
          null;

        if (!currentUser || !accessToken) {
          setUser(null);
          return;
        }

        /*
         * fetchUserAttributes() gives us the actual Cognito
         * attributes for federated users, including email.
         */
        let attributes = {};

        try {
          attributes = await fetchUserAttributes();
        } catch (attributeError) {
          console.warn(
            '[AKASHA] Unable to fetch Cognito user attributes:',
            attributeError
          );
        }

        if (!mounted) return;

        const idTokenPayload = session.tokens?.idToken?.payload || {};
        const accessTokenPayload = session.tokens?.accessToken?.payload || {};

        const email =
          attributes?.email ||
          idTokenPayload?.email ||
          accessTokenPayload?.email ||
          currentUser.signInDetails?.loginId ||
          null;

        const rawUsername = currentUser.username || '';
        const isGoogle = rawUsername.startsWith('google_');

        const name =
          attributes?.name ||
          idTokenPayload?.name ||
          attributes?.preferred_username ||
          (email ? email.split('@')[0] : null) ||
          (!isGoogle && rawUsername ? rawUsername : null) ||
          'Astronaut';

        const avatar =
          attributes?.picture ||
          idTokenPayload?.picture ||
          null;

        const provider = isGoogle ? 'Google' : 'Cognito';

        setUser({
          id: currentUser.userId,
          email,
          name,
          avatar,
          provider
        });
      } catch (err) {
        /*
         * No Cognito session is normal when logged out.
         */
        if (
          err?.name !== 'UserNotFoundException' &&
          err?.name !== 'NotAuthorizedException'
        ) {
          console.warn(
            '[AKASHA] Cognito session check:',
            err
          );
        }

        if (mounted) {
          setUser(null);
        }
      } finally {
        if (mounted) {
          setAuthLoading(false);
        }
      }
    };

    initializeAuth();

    const unsubscribe = Hub.listen('auth', ({ payload }) => {
      const { event } = payload || {};
      if (
        event === 'signedIn' ||
        event === 'signInWithRedirect' ||
        event === 'customOAuthState'
      ) {
        initializeAuth();
      } else if (event === 'signedOut') {
        if (mounted) {
          setUser(null);
          setAuthLoading(false);
        }
      } else if (event === 'signInWithRedirect_failure') {
        console.error('[AKASHA] OAuth redirect failure:', payload?.data);
        if (mounted) {
          setAuthLoading(false);
        }
      }
    });

    return () => {
      mounted = false;
      unsubscribe();
    };
  }, []);

  /* ──────────────────────────────────────────────────────────────
   * Session storage
   *
   * Temporary local implementation.
   * This will move to FastAPI + DynamoDB.
   * ────────────────────────────────────────────────────────────── */
  useEffect(() => {
    if (!user) {
      setSessions([]);
      setActiveSessionId(null);
      return;
    }

    const storageKey =
      `akasha_sessions_${user.id}`;

    try {
      const saved =
        localStorage.getItem(storageKey);

      if (saved) {
        const parsed = JSON.parse(saved);

        if (
          Array.isArray(parsed) &&
          parsed.length > 0
        ) {
          setSessions(parsed);

          setActiveSessionId(prev =>
            prev &&
            parsed.some(
              session => session.id === prev
            )
              ? prev
              : parsed[0].id
          );

          return;
        }
      }
    } catch (err) {
      console.warn(
        '[AKASHA] Failed to restore local sessions:',
        err
      );
    }

    const initialSession = {
      id: `session-${Date.now()}`,
      title: 'Satellite Imagery Overview',
      messages: [],
      updatedAt: Date.now()
    };

    setSessions([initialSession]);
    setActiveSessionId(initialSession.id);

    try {
      localStorage.setItem(
        storageKey,
        JSON.stringify([initialSession])
      );
    } catch (_) {}
  }, [user]);

  /* ──────────────────────────────────────────────────────────────
   * Persist sessions locally
   * ────────────────────────────────────────────────────────────── */
  useEffect(() => {
    if (!user || sessions.length === 0) {
      return;
    }

    try {
      localStorage.setItem(
        `akasha_sessions_${user.id}`,
        JSON.stringify(sessions)
      );
    } catch (err) {
      console.warn(
        '[AKASHA] Failed to persist local sessions:',
        err
      );
    }
  }, [user, sessions]);

  const activeSession =
    sessions.find(
      session => session.id === activeSessionId
    ) || sessions[0];

  /* ──────────────────────────────────────────────────────────────
   * Create new chat
   * ────────────────────────────────────────────────────────────── */
  const handleNewChat = async () => {
    const newId =
      `session-${Date.now()}`;

    const newSession = {
      id: newId,
      title:
        `Analysis Thread #${sessions.length + 1}`,
      messages: [],
      updatedAt: Date.now()
    };

    setSessions(prev => [
      newSession,
      ...prev
    ]);

    setActiveSessionId(newId);
  };

  /* ──────────────────────────────────────────────────────────────
   * Delete chat session
   * ────────────────────────────────────────────────────────────── */
  const handleDeleteSession = async id => {
    const remaining =
      sessions.filter(
        session => session.id !== id
      );

    setSessions(remaining);

    if (activeSessionId === id) {
      setActiveSessionId(
        remaining.length > 0
          ? remaining[0].id
          : null
      );
    }
  };

  /* ──────────────────────────────────────────────────────────────
   * Update session messages
   * ────────────────────────────────────────────────────────────── */
  const handleUpdateSessionMessages =
    async newMessages => {
      if (!activeSessionId) return;

      let updatedTitle =
        activeSession?.title ||
        'New Analysis Thread';

      /*
       * Automatically title a new session using
       * its first user message.
       */
      if (
        (!activeSession?.messages ||
          activeSession.messages.length === 0) &&
        newMessages.length > 0
      ) {
        const firstUserMsg =
          newMessages.find(
            message =>
              message.sender === 'user'
          );

        if (firstUserMsg?.text) {
          updatedTitle =
            firstUserMsg.text.length > 25
              ? `${firstUserMsg.text.substring(
                  0,
                  25
                )}...`
              : firstUserMsg.text;
        }
      }

      const updatedSession = {
        id: activeSessionId,
        title: updatedTitle,
        messages: newMessages,
        updatedAt: Date.now()
      };

      setSessions(prev =>
        prev.map(session =>
          session.id === activeSessionId
            ? updatedSession
            : session
        )
      );
    };

  /* ──────────────────────────────────────────────────────────────
   * Login
   * ────────────────────────────────────────────────────────────── */
  const handleLogin = userObj => {
    setUser(userObj);
  };

  /* ──────────────────────────────────────────────────────────────
   * Logout
   * ────────────────────────────────────────────────────────────── */
  const handleLogout = async () => {
    try {
      /*
       * Only real Cognito sessions should call Cognito logout.
       * Demo sessions are local.
       */
      if (
        user?.provider === 'Google' ||
        user?.provider === 'Cognito'
      ) {
        await cognitoSignOut();
      }
    } catch (err) {
      console.error(
        '[AKASHA] Cognito logout error:',
        err
      );
    }

    setUser(null);
    setSessions([]);
    setActiveSessionId(null);
  };

  /* ──────────────────────────────────────────────────────────────
   * Loading screen
   * ────────────────────────────────────────────────────────────── */
  if (authLoading) {
    return (
      <div
        style={{
          position: 'fixed',
          inset: 0,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          background: 'var(--bg-primary)',
          flexDirection: 'column',
          gap: '14px'
        }}
      >
        <div
          className="spinner"
          style={{
            width: 24,
            height: 24
          }}
          aria-label="Loading"
        />

        <p
          style={{
            color: 'var(--text-muted)',
            fontSize: '0.82rem',
            letterSpacing: '-0.01em'
          }}
        >
          Loading session…
        </p>
      </div>
    );
  }

  /* ──────────────────────────────────────────────────────────────
   * Login screen
   * ────────────────────────────────────────────────────────────── */
  if (!user) {
    return (
      <Login onLogin={handleLogin} />
    );
  }
  

  /* ──────────────────────────────────────────────────────────────
   * Main application
   * ────────────────────────────────────────────────────────────── */
  return (
    <div className="app-container">
      <Sidebar
        sessions={sessions}
        activeSessionId={activeSessionId}
        onSelectSession={setActiveSessionId}
        onNewChat={handleNewChat}
        onDeleteSession={handleDeleteSession}
        selectedFiles={selectedFiles}
        onFileSelect={setSelectedFiles}
        user={user}
        onLogout={handleLogout}
        theme={theme}
        onToggleTheme={toggleTheme}
      />

      <ChatInterface
        selectedFiles={selectedFiles}
        onFileSelect={setSelectedFiles}
        activeSession={activeSession}
        onUpdateSessionMessages={
          handleUpdateSessionMessages
        }
        user={user}
        draftQuery={currentDraft.query}
        onDraftQueryChange={setDraftQuery}
        onClearDraft={clearDraft}
      />
    </div>
  );
}

export default App;