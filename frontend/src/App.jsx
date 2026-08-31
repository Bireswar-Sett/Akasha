import React, { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import ChatInterface from './components/ChatInterface';
import Login from './components/Login';
import { auth, signOut } from './firebaseClient';
import { onAuthStateChanged } from 'firebase/auth';
import './index.css';

function App() {
  /* ── Auth state ── */
  const [user, setUser] = useState(null);
  const [authLoading, setAuthLoading] = useState(true); // prevents flash of login on refresh

  /* On mount: listen for Firebase Auth state changes */
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
      }
      setAuthLoading(false);
    });

    return () => unsubscribe();
  }, []);

  /* Called by Login component after successful login (real or demo fallback) */
  const handleLogin = (userObj) => {
    setUser(userObj);
  };

  /* Called by Sidebar logout button */
  const handleLogout = async () => {
    try {
      await signOut(auth);
    } catch (err) {
      console.error('Logout error:', err);
    }
    setUser(null);
  };

  /* ── Chat session state ── */
  const [sessions, setSessions] = useState([
    {
      id: 'session-1',
      title: 'Satellite Imagery Overview',
      messages: []
    }
  ]);
  const [activeSessionId, setActiveSessionId] = useState('session-1');
  const [selectedFiles, setSelectedFiles] = useState([]);

  const activeSession = sessions.find(s => s.id === activeSessionId) || sessions[0];

  const handleNewChat = () => {
    const newId = `session-${Date.now()}`;
    const newSession = {
      id: newId,
      title: `Analysis Thread #${sessions.length + 1}`,
      messages: []
    };
    setSessions([newSession, ...sessions]);
    setActiveSessionId(newId);
  };

  const handleDeleteSession = (id) => {
    const remaining = sessions.filter(s => s.id !== id);
    setSessions(remaining);
    if (activeSessionId === id && remaining.length > 0) {
      setActiveSessionId(remaining[0].id);
    }
  };

  const handleClearAll = () => {
    setSessions([]);
    handleNewChat();
  };

  const handleUpdateSessionMessages = (newMessages) => {
    setSessions(prevSessions => prevSessions.map(session => {
      if (session.id === activeSessionId) {
        // Auto-title session based on first user query if still generic
        let updatedTitle = session.title;
        if (session.messages.length === 0 && newMessages.length > 0) {
          const firstUserMsg = newMessages.find(m => m.sender === 'user');
          if (firstUserMsg) {
            updatedTitle = firstUserMsg.text.length > 25 
              ? `${firstUserMsg.text.substring(0, 25)}...` 
              : firstUserMsg.text;
          }
        }
        return {
          ...session,
          title: updatedTitle,
          messages: newMessages
        };
      }
      return session;
    }));
  };

  /* ── Render ── */

  // Waiting for Firebase to restore session – show a minimal loading screen
  if (authLoading) {
    return (
      <div style={{
        position: 'fixed',
        inset: 0,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: '#060611',
        flexDirection: 'column',
        gap: '16px'
      }}>
        <div className="spinner" style={{ width: 32, height: 32 }} aria-label="Loading" />
        <p style={{ color: '#9E9EB9', fontSize: '0.85rem', letterSpacing: '0.5px' }}>
          Establishing satellite uplink…
        </p>
      </div>
    );
  }

  // Show login if no authenticated user
  if (!user) {
    return <Login onLogin={handleLogin} />;
  }

  // Main dashboard
  return (
    <div className="app-container">
      <Sidebar
        sessions={sessions}
        activeSessionId={activeSessionId}
        onSelectSession={setActiveSessionId}
        onNewChat={handleNewChat}
        onDeleteSession={handleDeleteSession}
        onClearAll={handleClearAll}
        selectedFiles={selectedFiles}
        onFileSelect={setSelectedFiles}
        user={user}
        onLogout={handleLogout}
      />
      
      <ChatInterface
        selectedFiles={selectedFiles}
        onFileSelect={setSelectedFiles}
        activeSession={activeSession}
        onUpdateSessionMessages={handleUpdateSessionMessages}
      />
    </div>
  );
}

export default App;
