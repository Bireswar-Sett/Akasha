import React, { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import ChatInterface from './components/ChatInterface';
import Login from './components/Login';
import { auth, db, signOut, isDemoMode } from './firebaseClient';
import { onAuthStateChanged } from 'firebase/auth';
import { 
  collection, 
  doc, 
  setDoc, 
  deleteDoc, 
  onSnapshot, 
  query, 
  orderBy 
} from 'firebase/firestore';
import './index.css';

function App() {
  /* ── Theme state: 'dark' | 'light' ── */
  const [theme, setTheme] = useState(() => {
    try {
      return localStorage.getItem('akasha_theme') || 'dark';
    } catch (_) {
      return 'dark';
    }
  });

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    try {
      localStorage.setItem('akasha_theme', theme);
    } catch (_) {}
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prev => prev === 'dark' ? 'light' : 'dark');
  };

  /* ── Auth state ── */
  const [user, setUser] = useState(null);
  const [authLoading, setAuthLoading] = useState(true);

  /* ── Chat sessions state ── */
  const [sessions, setSessions] = useState([]);
  const [activeSessionId, setActiveSessionId] = useState(null);

  /* ── Per-session draft cache: { [sessionId]: { query: '', files: [] } } ── */
  // Restore query drafts from localStorage on mount (files can't be restored across refresh)
  const [drafts, setDrafts] = useState(() => {
    try {
      const saved = localStorage.getItem('akasha_drafts');
      if (saved) {
        const parsed = JSON.parse(saved);
        // Restore only query strings; files can't survive a page refresh
        const restored = {};
        Object.entries(parsed).forEach(([sid, d]) => {
          restored[sid] = { query: d.query || '', files: [] };
        });
        return restored;
      }
    } catch (_) {}
    return {};
  });

  /* Get / set the draft for the current active session */
  const currentDraft = drafts[activeSessionId] || { query: '', files: [] };
  const selectedFiles = currentDraft.files;

  const setSelectedFiles = (files) => {
    setDrafts(prev => ({
      ...prev,
      [activeSessionId]: { ...(prev[activeSessionId] || { query: '', files: [] }), files }
    }));
  };

  const setDraftQuery = (q) => {
    setDrafts(prev => ({
      ...prev,
      [activeSessionId]: { ...(prev[activeSessionId] || { query: '', files: [] }), query: q }
    }));
  };

  const clearDraft = () => {
    setDrafts(prev => ({
      ...prev,
      [activeSessionId]: { query: '', files: [] }
    }));
  };

  /* Persist draft queries (not files) to localStorage on every change */
  useEffect(() => {
    try {
      const toSave = {};
      Object.entries(drafts).forEach(([sid, d]) => {
        if (d.query) toSave[sid] = { query: d.query };
      });
      localStorage.setItem('akasha_drafts', JSON.stringify(toSave));
    } catch (_) {}
  }, [drafts]);

  /* Listen for Firebase Auth state changes */
  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (firebaseUser) => {
      if (firebaseUser) {
        setUser({
          id: firebaseUser.uid,
          email: firebaseUser.email,
          name: firebaseUser.displayName || firebaseUser.email?.split('@')[0] || 'Astronaut',
          avatar: firebaseUser.photoURL || null,
          provider: firebaseUser.providerData[0]?.providerId || 'email',
        });
      } else {
        setUser(null);
        setSessions([]);
        setActiveSessionId(null);
      }
      setAuthLoading(false);
    });

    return () => unsubscribe();
  }, []);

  /* Sync Chat Sessions from/to Firestore when user logs in */
  useEffect(() => {
    if (!user) return;

    if (isDemoMode) {
      // Demo fallback: local session
      const defaultSession = {
        id: 'session-1',
        title: 'Satellite Imagery Overview',
        messages: [],
        updatedAt: Date.now()
      };
      setSessions([defaultSession]);
      setActiveSessionId('session-1');
      return;
    }

    // Real Firestore sync per user: users/{userId}/sessions
    const sessionsRef = collection(db, 'users', user.id, 'sessions');
    const q = query(sessionsRef, orderBy('updatedAt', 'desc'));

    const unsubscribe = onSnapshot(q, (snapshot) => {
      const docs = snapshot.docs.map(d => ({ id: d.id, ...d.data() }));
      if (docs.length > 0) {
        setSessions(docs);
        setActiveSessionId(prev => prev && docs.some(s => s.id === prev) ? prev : docs[0].id);
      } else {
        // First time user login: create an initial session in Firestore
        const initId = `session-${Date.now()}`;
        const initSession = {
          id: initId,
          title: 'Satellite Imagery Overview',
          messages: [],
          updatedAt: Date.now()
        };
        setDoc(doc(db, 'users', user.id, 'sessions', initId), initSession);
        setSessions([initSession]);
        setActiveSessionId(initId);
      }
    }, (err) => {
      console.warn('Firestore sync warning:', err);
      // Fallback if rules or collection offline
      const fallback = {
        id: 'session-1',
        title: 'Satellite Imagery Overview',
        messages: [],
        updatedAt: Date.now()
      };
      setSessions([fallback]);
      setActiveSessionId('session-1');
    });

    return () => unsubscribe();
  }, [user]);

  const activeSession = sessions.find(s => s.id === activeSessionId) || sessions[0];

  /* Create new chat session */
  const handleNewChat = async () => {
    const newId = `session-${Date.now()}`;
    const newSession = {
      id: newId,
      title: `Analysis Thread #${sessions.length + 1}`,
      messages: [],
      updatedAt: Date.now()
    };

    if (user && !isDemoMode) {
      try {
        await setDoc(doc(db, 'users', user.id, 'sessions', newId), newSession);
      } catch (err) {
        console.error('Error saving new session to Firestore:', err);
      }
    } else {
      setSessions(prev => [newSession, ...prev]);
    }
    setActiveSessionId(newId);
  };

  /* Delete single chat session */
  const handleDeleteSession = async (id) => {
    const remaining = sessions.filter(s => s.id !== id);
    if (user && !isDemoMode) {
      try {
        await deleteDoc(doc(db, 'users', user.id, 'sessions', id));
      } catch (err) {
        console.error('Error deleting session from Firestore:', err);
      }
    } else {
      setSessions(remaining);
    }
    if (activeSessionId === id && remaining.length > 0) {
      setActiveSessionId(remaining[0].id);
    }
  };


  /* Update messages for the active session and persist to Firestore */
  const handleUpdateSessionMessages = async (newMessages) => {
    if (!activeSessionId) return;

    let updatedTitle = activeSession?.title || 'New Analysis Thread';
    if ((!activeSession?.messages || activeSession.messages.length === 0) && newMessages.length > 0) {
      const firstUserMsg = newMessages.find(m => m.sender === 'user');
      if (firstUserMsg && firstUserMsg.text) {
        updatedTitle = firstUserMsg.text.length > 25 
          ? `${firstUserMsg.text.substring(0, 25)}...` 
          : firstUserMsg.text;
      }
    }

    const updatedSession = {
      id: activeSessionId,
      title: updatedTitle,
      messages: newMessages,
      updatedAt: Date.now()
    };

    // Sanitize object to remove non-serializable fields (like File objects or undefined) before Firestore setDoc
    const cleanSession = JSON.parse(JSON.stringify(updatedSession));

    if (user && !isDemoMode) {
      try {
        await setDoc(doc(db, 'users', user.id, 'sessions', activeSessionId), cleanSession, { merge: true });
      } catch (err) {
        console.error('Error persisting messages to Firestore:', err);
      }
    } else {
      setSessions(prev => prev.map(s => s.id === activeSessionId ? updatedSession : s));
    }
  };

  const handleLogin = (userObj) => {
    setUser(userObj);
  };

  const handleLogout = async () => {
    try {
      await signOut(auth);
    } catch (err) {
      console.error('Logout error:', err);
    }
    setUser(null);
  };

  if (authLoading) {
    return (
      <div style={{
        position: 'fixed',
        inset: 0,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'var(--bg-primary)',
        flexDirection: 'column',
        gap: '14px'
      }}>
        <div className="spinner" style={{ width: 24, height: 24 }} aria-label="Loading" />
        <p style={{ color: 'var(--text-muted)', fontSize: '0.82rem', letterSpacing: '-0.01em' }}>
          Loading session…
        </p>
      </div>
    );
  }

  if (!user) {
    return <Login onLogin={handleLogin} />;
  }

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
        onUpdateSessionMessages={handleUpdateSessionMessages}
        user={user}
        draftQuery={currentDraft.query}
        onDraftQueryChange={setDraftQuery}
        onClearDraft={clearDraft}
      />
    </div>
  );
}

export default App;
