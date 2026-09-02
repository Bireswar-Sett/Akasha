import React, { useState, useEffect, useRef } from 'react';
import {
  Plus,
  MessageSquare,
  UploadCloud,
  Trash2,
  ChevronLeft,
  ChevronRight,
  LogOut,
  Settings,
  Sun,
  Moon,
  User as UserIcon,
  X,
  ShieldCheck
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
  selectedFiles,
  onFileSelect,
  user,
  onLogout,
  theme = 'dark',
  onToggleTheme
}) => {
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [activeTab, setActiveTab] = useState('chats'); // 'chats' | 'imagery'
  const [userImagery, setUserImagery] = useState([]);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isProfileOpen, setIsProfileOpen] = useState(false);
  const settingsRef = useRef(null);

  /* Close settings dropdown when clicking outside */
  useEffect(() => {
    const handleOutsideClick = (e) => {
      if (settingsRef.current && !settingsRef.current.contains(e.target)) {
        setIsSettingsOpen(false);
      }
    };
    if (isSettingsOpen) {
      document.addEventListener('mousedown', handleOutsideClick);
    }
    return () => document.removeEventListener('mousedown', handleOutsideClick);
  }, [isSettingsOpen]);

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
    <>
      <aside style={{
        width: isCollapsed ? '68px' : '280px',
        minWidth: isCollapsed ? '68px' : '280px',
        backgroundColor: 'var(--bg-sidebar)',
        borderRight: '1px solid var(--border-subtle)',
        display: 'flex',
        flexDirection: 'column',
        transition: 'width 0.22s cubic-bezier(0.16, 1, 0.3, 1), min-width 0.22s cubic-bezier(0.16, 1, 0.3, 1), background-color 0.2s ease',
        height: '100vh',
        zIndex: 10,
        position: 'relative'
      }}>
        {/* Sidebar Header */}
        <div style={{
          padding: isCollapsed ? '1rem 0.5rem' : '1.1rem 1rem',
          display: 'flex',
          alignItems: 'center',
          justifyContent: isCollapsed ? 'center' : 'space-between',
          borderBottom: '1px solid var(--border-subtle)'
        }}>
          {!isCollapsed && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <img
                src={logoSrc}
                alt="AKASHA"
                style={{
                  height: '30px',
                  width: 'auto',
                  borderRadius: '6px',
                  objectFit: 'contain'
                }}
              />
              <div>
                <div style={{ fontSize: '1.05rem', fontWeight: 700, letterSpacing: '-0.02em', color: 'var(--text-primary)' }}>
                  AKASHA
                </div>
                <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)', letterSpacing: '0.06em', textTransform: 'uppercase', fontWeight: 600 }}>
                  Geospatial AI
                </span>
              </div>
            </div>
          )}
          <button
            onClick={() => setIsCollapsed(!isCollapsed)}
            className="sidebar-action-btn"
            title={isCollapsed ? "Expand Sidebar" : "Collapse Sidebar"}
            aria-label={isCollapsed ? "Expand Sidebar" : "Collapse Sidebar"}
          >
            {isCollapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
          </button>
        </div>

        {/* Action: New Analysis Button */}
        <div style={{ padding: '0.85rem 0.75rem 0.4rem 0.75rem' }}>
          <button
            onClick={onNewChat}
            className="btn-primary"
            style={{
              width: '100%',
              justifyContent: isCollapsed ? 'center' : 'flex-start',
              padding: isCollapsed ? '10px' : '9px 12px',
              borderRadius: '8px',
              fontSize: '0.84rem'
            }}
            title="New Analysis"
          >
            <Plus size={16} />
            {!isCollapsed && <span>New Analysis</span>}
          </button>
        </div>

        {/* Navigation Tabs (if not collapsed) */}
        {!isCollapsed && (
          <div style={{
            display: 'flex',
            margin: '0.4rem 0.75rem',
            background: 'var(--bg-secondary)',
            border: '1px solid var(--border-subtle)',
            borderRadius: '8px',
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
                  type="button"
                  style={{
                    flex: 1,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: '6px',
                    padding: '5px 0',
                    fontSize: '0.78rem',
                    fontWeight: active ? 600 : 400,
                    color: active ? 'var(--text-primary)' : 'var(--text-muted)',
                    background: active ? 'var(--bg-card)' : 'transparent',
                    border: 'none',
                    borderRadius: '5px',
                    cursor: 'pointer',
                    boxShadow: active ? '0 1px 3px rgba(0,0,0,0.1)' : 'none',
                    transition: 'all 0.15s ease'
                  }}
                >
                  <Icon size={13} color={active ? 'var(--text-primary)' : 'var(--text-muted)'} />
                  {tab.label}
                </button>
              );
            })}
          </div>
        )}

        {/* Main Scrollable Section */}
        <div style={{
          flex: 1,
          overflowY: 'auto',
          padding: '0.4rem 0.75rem',
          display: 'flex',
          flexDirection: 'column',
          gap: '6px'
        }}>
          {isCollapsed ? (
            /* Collapsed Icons view */
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', alignItems: 'center', marginTop: '6px' }}>
              {sessions.map(s => {
                const isActive = s.id === activeSessionId;
                return (
                  <button
                    key={s.id}
                    onClick={() => onSelectSession(s.id)}
                    style={{
                      width: '38px',
                      height: '38px',
                      borderRadius: '8px',
                      background: isActive ? 'var(--bg-card-hover)' : 'transparent',
                      border: isActive ? '1px solid var(--border-medium)' : '1px solid transparent',
                      color: isActive ? 'var(--text-primary)' : 'var(--text-muted)',
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      transition: 'all 0.15s ease'
                    }}
                    title={s.title}
                  >
                    <MessageSquare size={16} />
                  </button>
                );
              })}
            </div>
          ) : (
            <>
              {activeTab === 'chats' && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                  <span style={{ fontSize: '0.68rem', fontWeight: 600, color: 'var(--text-muted)', padding: '4px 6px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                    Recent Analyses
                  </span>
                  {sessions.length === 0 ? (
                    <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', padding: '12px 8px', textAlign: 'center' }}>
                      No sessions yet.
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
                            padding: '8px 10px',
                            borderRadius: '8px',
                            background: isActive ? 'var(--bg-card-hover)' : 'transparent',
                            border: isActive ? '1px solid var(--border-subtle)' : '1px solid transparent',
                            cursor: 'pointer',
                            transition: 'all 0.15s ease'
                          }}
                          onMouseEnter={e => {
                            if (!isActive) e.currentTarget.style.background = 'var(--glass-bg-hover)';
                          }}
                          onMouseLeave={e => {
                            if (!isActive) e.currentTarget.style.background = 'transparent';
                          }}
                        >
                          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', overflow: 'hidden' }}>
                            <MessageSquare size={14} color={isActive ? 'var(--text-primary)' : 'var(--text-muted)'} style={{ flexShrink: 0 }} />
                            <span style={{
                              fontSize: '0.82rem',
                              color: isActive ? 'var(--text-primary)' : 'var(--text-secondary)',
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
                              color: 'var(--text-muted)',
                              cursor: 'pointer',
                              padding: '3px',
                              display: 'flex',
                              borderRadius: '4px',
                              transition: 'color 0.15s'
                            }}
                            onMouseEnter={(e) => e.currentTarget.style.color = '#ef4444'}
                            onMouseLeave={(e) => e.currentTarget.style.color = 'var(--text-muted)'}
                            title="Delete Session"
                          >
                            <Trash2 size={13} />
                          </button>
                        </div>
                      );
                    })
                  )}
                </div>
              )}

              {activeTab === 'imagery' && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                  <span style={{ fontSize: '0.68rem', fontWeight: 600, color: 'var(--text-muted)', padding: '4px 6px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                    Uploaded Files ({userImagery.length})
                  </span>

                  <div
                    {...getRootProps()}
                    style={{
                      border: `1px dashed ${isDragActive ? 'var(--accent-primary)' : 'var(--border-medium)'}`,
                      borderRadius: '8px',
                      padding: '0.85rem',
                      textAlign: 'center',
                      cursor: 'pointer',
                      background: isDragActive ? 'var(--glass-bg-hover)' : 'transparent',
                      transition: 'all 0.18s ease'
                    }}
                  >
                    <input {...getInputProps()} />
                    <UploadCloud size={20} color="var(--text-secondary)" style={{ margin: '0 auto 4px' }} />
                    <p style={{ fontSize: '0.78rem', color: 'var(--text-primary)', fontWeight: 500 }}>
                      {isDragActive ? 'Drop imagery here' : 'Add Imagery File'}
                    </p>
                    <p style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '2px' }}>
                      GeoTIFF, TIFF, PNG, JPEG
                    </p>
                  </div>

                  {/* Stored Cloud Imagery */}
                  {userImagery.length === 0 ? (
                    <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', padding: '10px 6px', textAlign: 'center' }}>
                      No imagery uploaded yet.
                    </p>
                  ) : (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', marginTop: '4px' }}>
                      {userImagery.map(img => (
                        <a
                          key={img.id}
                          href={img.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          title="View image"
                          style={{
                            display: 'flex',
                            alignItems: 'center',
                            gap: '8px',
                            background: 'var(--bg-card)',
                            padding: '6px 8px',
                            borderRadius: '6px',
                            border: '1px solid var(--border-subtle)',
                            fontSize: '0.76rem',
                            textDecoration: 'none',
                            cursor: 'pointer',
                            transition: 'all 0.15s ease'
                          }}
                          onMouseEnter={e => {
                            e.currentTarget.style.background = 'var(--bg-card-hover)';
                            e.currentTarget.style.borderColor = 'var(--border-medium)';
                          }}
                          onMouseLeave={e => {
                            e.currentTarget.style.background = 'var(--bg-card)';
                            e.currentTarget.style.borderColor = 'var(--border-subtle)';
                          }}
                        >
                          <img
                            src={img.url}
                            alt={img.name}
                            style={{ width: '28px', height: '28px', borderRadius: '4px', objectFit: 'cover' }}
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

        {/* Footer / User Profile & Settings */}
        <div style={{
          padding: '0.75rem',
          borderTop: '1px solid var(--border-subtle)',
          display: 'flex',
          flexDirection: 'column',
          gap: '6px',
          position: 'relative'
        }}>
          {/* User Card */}
          {user && (
            isCollapsed ? (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '6px' }}>
                <button
                  className="sidebar-action-btn"
                  onClick={() => setIsSettingsOpen(!isSettingsOpen)}
                  title="Settings"
                  aria-label="Settings"
                >
                  <Settings size={16} />
                </button>
              </div>
            ) : (
              <div className="sidebar-user-card" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <div
                  onClick={() => setIsProfileOpen(true)}
                  style={{ display: 'flex', alignItems: 'center', gap: '10px', overflow: 'hidden', cursor: 'pointer', flex: 1 }}
                  title="Click to view profile"
                >
                  {user.avatar ? (
                    <img src={user.avatar} alt={user.name} className="sidebar-user-avatar" />
                  ) : (
                    <div className="sidebar-user-avatar-fallback" aria-hidden="true">
                      {(user.name || user.email || 'U')[0].toUpperCase()}
                    </div>
                  )}
                  <div className="sidebar-user-info">
                    <div className="sidebar-user-name">{user.name || 'User'}</div>
                    <div className="sidebar-user-sub">{user.email || user.provider}</div>
                  </div>
                </div>

                {/* Settings Dropdown Button */}
                <div className="settings-menu-container" ref={settingsRef}>
                  <button
                    className="sidebar-action-btn"
                    onClick={() => setIsSettingsOpen(!isSettingsOpen)}
                    title="Settings & Theme"
                    aria-label="Settings & Theme"
                    aria-expanded={isSettingsOpen}
                  >
                    <Settings size={15} />
                  </button>

                  {/* Settings Dropdown Menu */}
                  {isSettingsOpen && (
                    <div className="settings-dropdown" role="menu">
                      {/* Theme Switcher Item */}
                      <button
                        className="settings-menu-item"
                        onClick={() => {
                          if (onToggleTheme) onToggleTheme();
                        }}
                        role="menuitem"
                      >
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                          {theme === 'dark' ? <Moon size={15} /> : <Sun size={15} />}
                          <span>Theme</span>
                        </div>
                        <span className="theme-badge">{theme === 'dark' ? 'Dark' : 'Light'}</span>
                      </button>

                      {/* View Profile Item */}
                      <button
                        className="settings-menu-item"
                        onClick={() => {
                          setIsProfileOpen(true);
                          setIsSettingsOpen(false);
                        }}
                        role="menuitem"
                      >
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <UserIcon size={15} />
                          <span>View Profile</span>
                        </div>
                      </button>

                      <div className="settings-menu-divider" />

                      {/* Sign Out Item */}
                      <button
                        className="settings-menu-item"
                        onClick={() => {
                          setIsSettingsOpen(false);
                          if (onLogout) onLogout();
                        }}
                        style={{ color: '#ef4444' }}
                        role="menuitem"
                      >
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <LogOut size={15} />
                          <span>Sign Out</span>
                        </div>
                      </button>
                    </div>
                  )}
                </div>
              </div>
            )
          )}

          {!isCollapsed && (
            <div style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: '2px 4px',
              color: 'var(--text-muted)',
              fontSize: '0.7rem'
            }}>
              <span>AKASHA v1.0</span>
            </div>
          )}
        </div>
      </aside>

      {/* View Profile Modal */}
      {isProfileOpen && (
        <div
          className="profile-modal-overlay"
          onClick={() => setIsProfileOpen(false)}
          role="dialog"
          aria-modal="true"
          aria-labelledby="profile-modal-title"
        >
          <div
            className="profile-modal-card"
            onClick={e => e.stopPropagation()}
          >
            {/* Header */}
            <div className="profile-modal-header">
              <h3 id="profile-modal-title" style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                Operator Profile
              </h3>
              <button
                onClick={() => setIsProfileOpen(false)}
                className="sidebar-action-btn"
                aria-label="Close profile"
              >
                <X size={16} />
              </button>
            </div>

            {/* User summary card */}
            <div className="profile-user-summary">
              {user?.avatar ? (
                <img src={user.avatar} alt={user.name} className="profile-avatar-large" />
              ) : (
                <div className="profile-avatar-fallback-large">
                  {(user?.name || user?.email || 'U')[0].toUpperCase()}
                </div>
              )}
              <div style={{ overflow: 'hidden' }}>
                <div style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                  {user?.name || 'Operator'}
                </div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                  {user?.email || 'No email provided'}
                </div>
                <div style={{ display: 'inline-flex', alignItems: 'center', gap: '4px', fontSize: '0.72rem', color: '#22c55e', marginTop: '4px' }}>
                  <ShieldCheck size={13} /> Active Session
                </div>
              </div>
            </div>

            {/* Details Grid */}
            <div className="profile-details-grid">
              <div className="profile-detail-card">
                <div className="profile-detail-label">Auth Provider</div>
                <div className="profile-detail-value" style={{ textTransform: 'capitalize' }}>
                  {user?.provider || 'Email / Password'}
                </div>
              </div>

              <div className="profile-detail-card">
                <div className="profile-detail-label">Active Theme</div>
                <div className="profile-detail-value" style={{ textTransform: 'capitalize' }}>
                  {theme} Mode
                </div>
              </div>

              <div className="profile-detail-card">
                <div className="profile-detail-label">Analyses Count</div>
                <div className="profile-detail-value">
                  {sessions.length} Sessions
                </div>
              </div>

              <div className="profile-detail-card">
                <div className="profile-detail-label">Imagery Files</div>
                <div className="profile-detail-value">
                  {userImagery.length} Uploads
                </div>
              </div>
            </div>

            {/* Account UID */}
            {user?.id && (
              <div className="profile-detail-card">
                <div className="profile-detail-label">Account UID</div>
                <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', fontFamily: 'monospace', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                  {user.id}
                </div>
              </div>
            )}

            {/* Footer action */}
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '8px', marginTop: '0.25rem' }}>
              <button
                className="btn-primary"
                onClick={() => setIsProfileOpen(false)}
                style={{ padding: '8px 18px', fontSize: '0.84rem' }}
              >
                Done
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default Sidebar;
