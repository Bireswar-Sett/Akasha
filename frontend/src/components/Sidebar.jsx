import React, { useState, useEffect } from 'react';
import { 
  Plus, 
  MessageSquare, 
  UploadCloud, 
  Cpu, 
  Trash2, 
  ChevronLeft, 
  ChevronRight, 
  Layers, 
  Settings,
  Image as ImageIcon,
  CheckCircle2,
  X,
  LogOut
} from 'lucide-react';
import { useDropzone } from 'react-dropzone';
import logoSrc from '../assets/logo.png';
import { db, isDemoMode } from '../firebaseClient';
import { collection, query, orderBy, onSnapshot } from 'firebase/firestore';

const Sidebar = ({ 
  sessions, 
  activeSessionId, 
  onSelectSession, 
  onNewChat, 
  onDeleteSession,
  onClearAll,
  selectedFiles,
  onFileSelect,
  user,
  onLogout
}) => {
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [activeTab, setActiveTab] = useState('chats'); // 'chats' | 'imagery'
  const [userImagery, setUserImagery] = useState([]);

  useEffect(() => {
    if (!user || isDemoMode) return;
    const imageryRef = collection(db, 'users', user.id, 'imagery');
    const q = query(imageryRef, orderBy('uploadedAt', 'desc'));
    const unsubscribe = onSnapshot(q, (snapshot) => {
      const items = snapshot.docs.map(d => ({ id: d.id, ...d.data() }));
      setUserImagery(items);
    }, (err) => console.warn('Firestore imagery list warning:', err));
    return () => unsubscribe();
  }, [user]);

  const onDrop = (acceptedFiles) => {
    if (acceptedFiles && acceptedFiles.length > 0) {
      onFileSelect([...selectedFiles, ...acceptedFiles]);
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/tiff': ['.tiff', '.tif'],
      'image/png': ['.png'],
      'image/jpeg': ['.jpeg', '.jpg']
    }
  });

  return (
    <aside style={{
      width: isCollapsed ? '72px' : '300px',
      minWidth: isCollapsed ? '72px' : '300px',
      backgroundColor: 'var(--bg-sidebar)',
      borderRight: '1px solid var(--glass-border)',
      display: 'flex',
      flexDirection: 'column',
      transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
      height: '100vh',
      zIndex: 10,
      position: 'relative'
    }}>
      {/* Sidebar Header */}
      <div style={{
        padding: '1.25rem 1rem',
        display: 'flex',
        alignItems: 'center',
        justifyContent: isCollapsed ? 'center' : 'space-between',
        borderBottom: '1px solid var(--glass-border)'
      }}>
        {!isCollapsed && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <img
              src={logoSrc}
              alt="AKASHA"
              style={{
                height: '38px',
                width: 'auto',
                filter: 'drop-shadow(0 0 8px rgba(108,99,255,0.5))'
              }}
            />
            <div>
              <h1 className="animate-shimmer" style={{ fontSize: '1.25rem', fontWeight: 700, lineHeight: 1.1 }}>
                AKASHA
              </h1>
              <span style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', letterSpacing: '0.5px' }}>
                SATELLITE INTELLIGENCE
              </span>
            </div>
          </div>
        )}
        <button
          onClick={() => setIsCollapsed(!isCollapsed)}
          style={{
            background: 'rgba(255,255,255,0.05)',
            border: '1px solid var(--glass-border)',
            borderRadius: '8px',
            color: 'var(--text-secondary)',
            cursor: 'pointer',
            padding: '6px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}
          title={isCollapsed ? "Expand Sidebar" : "Collapse Sidebar"}
        >
          {isCollapsed ? <ChevronRight size={18} /> : <ChevronLeft size={18} />}
        </button>
      </div>

      {/* Action: New Chat Button */}
      <div style={{ padding: '1rem 0.8rem 0.5rem 0.8rem' }}>
        <button
          onClick={onNewChat}
          className="glass-button"
          style={{
            width: '100%',
            justifyContent: isCollapsed ? 'center' : 'flex-start',
            padding: isCollapsed ? '12px' : '10px 14px',
            borderRadius: '12px'
          }}
          title="New Analysis Chat"
        >
          <Plus size={18} color="var(--accent-secondary)" />
          {!isCollapsed && <span>New Analysis</span>}
        </button>
      </div>

      {/* Navigation Tabs (if not collapsed) */}
      {!isCollapsed && (
        <div style={{
          display: 'flex',
          margin: '0.5rem 0.8rem',
          background: 'rgba(0, 0, 0, 0.25)',
          borderRadius: '10px',
          padding: '3px',
          gap: '2px'
        }}>
          {[
            { id: 'chats', label: 'History', icon: MessageSquare },
            { id: 'imagery', label: 'Imagery', icon: UploadCloud }
          ].map(tab => {
            const Icon = tab.icon;
            const active = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                style={{
                  flex: 1,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '6px',
                  padding: '6px 0',
                  fontSize: '0.78rem',
                  fontWeight: active ? 600 : 400,
                  color: active ? '#fff' : 'var(--text-secondary)',
                  background: active ? 'rgba(255,255,255,0.08)' : 'transparent',
                  border: 'none',
                  borderRadius: '7px',
                  cursor: 'pointer',
                  transition: 'all 0.2s ease'
                }}
              >
                <Icon size={14} color={active ? 'var(--accent-primary)' : 'var(--text-secondary)'} />
                {tab.label}
              </button>
            );
          })}
        </div>
      )}

      {/* Tab Contents / Main Scrollable Section */}
      <div style={{
        flex: 1,
        overflowY: 'auto',
        padding: '0.5rem 0.8rem',
        display: 'flex',
        flexDirection: 'column',
        gap: '8px'
      }}>
        {isCollapsed ? (
          /* Collapsed Icons view */
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', alignItems: 'center', marginTop: '10px' }}>
            {sessions.map(s => (
              <button
                key={s.id}
                onClick={() => onSelectSession(s.id)}
                style={{
                  width: '40px',
                  height: '40px',
                  borderRadius: '10px',
                  background: s.id === activeSessionId ? 'rgba(108, 99, 255, 0.2)' : 'rgba(255,255,255,0.04)',
                  border: s.id === activeSessionId ? '1px solid var(--accent-primary)' : '1px solid transparent',
                  color: s.id === activeSessionId ? 'var(--accent-secondary)' : 'var(--text-secondary)',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}
                title={s.title}
              >
                <MessageSquare size={18} />
              </button>
            ))}
          </div>
        ) : (
          <>
            {activeTab === 'chats' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                <span style={{ fontSize: '0.72rem', fontWeight: 600, color: 'var(--text-secondary)', padding: '4px 6px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                  Recent Conversations
                </span>
                {sessions.length === 0 ? (
                  <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', padding: '12px 8px', textAlign: 'center' }}>
                    No previous chats yet.
                  </p>
                ) : (
                  sessions.map(s => {
                    const isActive = s.id === activeSessionId;
                    return (
                      <div
                        key={s.id}
                        onClick={() => onSelectSession(s.id)}
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'space-between',
                          padding: '10px 12px',
                          borderRadius: '10px',
                          background: isActive ? 'rgba(108, 99, 255, 0.15)' : 'transparent',
                          border: isActive ? '1px solid rgba(108, 99, 255, 0.3)' : '1px solid transparent',
                          cursor: 'pointer',
                          transition: 'all 0.2s ease'
                        }}
                      >
                        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', overflow: 'hidden' }}>
                          <MessageSquare size={16} color={isActive ? 'var(--accent-secondary)' : 'var(--text-secondary)'} />
                          <span style={{
                            fontSize: '0.88rem',
                            color: isActive ? '#fff' : 'var(--text-secondary)',
                            fontWeight: isActive ? 500 : 400,
                            whiteSpace: 'nowrap',
                            overflow: 'hidden',
                            textOverflow: 'ellipsis'
                          }}>
                            {s.title}
                          </span>
                        </div>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            onDeleteSession(s.id);
                          }}
                          style={{
                            background: 'transparent',
                            border: 'none',
                            color: 'rgba(255,255,255,0.3)',
                            cursor: 'pointer',
                            padding: '4px',
                            display: 'flex',
                            borderRadius: '4px'
                          }}
                          onMouseEnter={(e) => e.currentTarget.style.color = '#ff4d4d'}
                          onMouseLeave={(e) => e.currentTarget.style.color = 'rgba(255,255,255,0.3)'}
                          title="Delete Session"
                        >
                          <Trash2 size={14} />
                        </button>
                      </div>
                    );
                  })
                )}
              </div>
            )}

            {activeTab === 'imagery' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                <span style={{ fontSize: '0.72rem', fontWeight: 600, color: 'var(--text-secondary)', padding: '4px 6px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                  Imagery Manager ({selectedFiles.length + userImagery.length})
                </span>

                <div
                  {...getRootProps()}
                  style={{
                    border: `2px dashed ${isDragActive ? 'var(--accent-primary)' : 'var(--glass-border)'}`,
                    borderRadius: '12px',
                    padding: '1rem',
                    textAlign: 'center',
                    cursor: 'pointer',
                    background: isDragActive ? 'rgba(108, 99, 255, 0.1)' : 'rgba(0, 0, 0, 0.2)',
                    transition: 'all 0.3s ease'
                  }}
                >
                  <input {...getInputProps()} />
                  <UploadCloud size={24} color="var(--accent-primary)" style={{ margin: '0 auto 6px' }} />
                  <p style={{ fontSize: '0.8rem', color: '#fff', fontWeight: 500 }}>
                    {isDragActive ? "Drop satellite imagery here" : "Upload Imagery"}
                  </p>
                  <p style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', marginTop: '2px' }}>
                    GeoTIFF, TIFF, PNG, JPEG
                  </p>
                </div>

                {/* Staged files */}
                {selectedFiles.map((file, idx) => (
                  <div key={idx} style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    background: 'rgba(108, 99, 255, 0.12)',
                    padding: '8px 10px',
                    borderRadius: '8px',
                    border: '1px solid rgba(108, 99, 255, 0.3)',
                    fontSize: '0.8rem'
                  }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', overflow: 'hidden' }}>
                      <ImageIcon size={16} color="var(--accent-secondary)" />
                      <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: '140px' }}>
                        {file.name}
                      </span>
                    </div>
                    <button
                      onClick={() => onFileSelect(selectedFiles.filter((_, i) => i !== idx))}
                      style={{ background: 'transparent', border: 'none', color: 'var(--text-secondary)', cursor: 'pointer' }}
                    >
                      <X size={14} />
                    </button>
                  </div>
                ))}

                {/* Persisted Storage Imagery */}
                {userImagery.length > 0 && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', marginTop: '6px' }}>
                    <span style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', textTransform: 'uppercase' }}>
                      Stored Cloud Imagery
                    </span>
                    {userImagery.map(img => (
                      <a
                        key={img.id}
                        href={img.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        title="Click to view full image in new tab"
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: '8px',
                          background: 'rgba(255,255,255,0.03)',
                          padding: '6px 8px',
                          borderRadius: '8px',
                          border: '1px solid var(--glass-border)',
                          fontSize: '0.78rem',
                          textDecoration: 'none',
                          cursor: 'pointer',
                          transition: 'all 0.2s ease'
                        }}
                        onMouseEnter={e => e.currentTarget.style.background = 'rgba(255,255,255,0.08)'}
                        onMouseLeave={e => e.currentTarget.style.background = 'rgba(255,255,255,0.03)'}
                      >
                        <img
                          src={img.url}
                          alt={img.name}
                          style={{ width: '32px', height: '32px', borderRadius: '6px', objectFit: 'cover' }}
                        />
                        <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1, color: 'var(--text-primary)' }}>
                          {img.name}
                        </span>
                      </a>
                    ))}
                  </div>
                )}
              </div>
            )}
          </>
        )}
      </div>

      {/* Footer / Utilities */}
      <div style={{
        padding: '0.8rem',
        borderTop: '1px solid var(--glass-border)',
        display: 'flex',
        flexDirection: 'column',
        gap: '6px'
      }}>
        {!isCollapsed && sessions.length > 0 && (
          <button
            onClick={onClearAll}
            style={{
              width: '100%',
              background: 'transparent',
              border: '1px solid rgba(255, 77, 77, 0.2)',
              borderRadius: '8px',
              padding: '6px',
              color: '#ff6b6b',
              fontSize: '0.78rem',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '6px'
            }}
          >
            <Trash2 size={14} /> Clear History
          </button>
        )}
        {/* User profile card */}
        {user && (
          isCollapsed ? (
            <div style={{ display: 'flex', justifyContent: 'center', padding: '6px 0' }}>
              <button
                className="sidebar-logout-btn"
                onClick={onLogout}
                title="Logout"
                aria-label="Logout"
              >
                <LogOut size={16} />
              </button>
            </div>
          ) : (
            <div className="sidebar-user-card">
              {user.avatar ? (
                <img src={user.avatar} alt={user.name} className="sidebar-user-avatar" />
              ) : (
                <div className="sidebar-user-avatar-fallback" aria-hidden="true">
                  {(user.name || user.email || '?')[0].toUpperCase()}
                </div>
              )}
              <div className="sidebar-user-info">
                <div className="sidebar-user-name">{user.name || 'Astronaut'}</div>
                <div className="sidebar-user-sub">{user.email || user.provider}</div>
              </div>
              <button
                className="sidebar-logout-btn"
                onClick={onLogout}
                title="Logout"
                aria-label="Logout"
              >
                <LogOut size={15} />
              </button>
            </div>
          )
        )}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: isCollapsed ? 'center' : 'space-between',
          padding: '6px',
          color: 'var(--text-secondary)',
          fontSize: '0.78rem'
        }}>
          {!isCollapsed && <span>AKASHA v1.0 • Qwen Router</span>}
          <Settings size={16} style={{ cursor: 'pointer' }} />
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;
