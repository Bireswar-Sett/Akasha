import React, { useState, useRef, useEffect } from 'react';
import { 
  Send, 
  Paperclip, 
  X, 
  Bot, 
  User, 
  Image as ImageIcon,
  ArrowUpRight
} from 'lucide-react';
import axios from 'axios';
import logoSrc from '../assets/logo.png';
import { storage, db, auth, isDemoMode } from '../firebaseClient';
import { ref, uploadBytes } from 'firebase/storage';
import { collection, addDoc } from 'firebase/firestore';

const ChatInterface = ({ 
  selectedFiles, 
  onFileSelect, 
  activeSession, 
  onUpdateSessionMessages,
  user,
  draftQuery,
  onDraftQueryChange,
  onClearDraft
}) => {
  const query = draftQuery || '';
  const setQuery = onDraftQueryChange;
  const [isProcessing, setIsProcessing] = useState(false);
  const [previewImage, setPreviewImage] = useState(null);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [activeSession?.messages, isProcessing]);

  const handleSend = async (overrideQuery = null) => {
    let textToSend = overrideQuery || query;
    if ((!textToSend || !textToSend.trim()) && selectedFiles && selectedFiles.length > 0) {
      textToSend = 'Analyze attached satellite imagery.';
    }
    if (!textToSend || !textToSend.trim() || isProcessing) return;

    const filesToUpload = selectedFiles ? [...selectedFiles] : [];
    if (onClearDraft) onClearDraft();
    setIsProcessing(true);

    // Step 1: User message
    const userMessage = {
      id: Date.now().toString(),
      sender: 'user',
      text: textToSend,
      attachments: [],
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };
    const messagesWithUser = [...(activeSession?.messages || []), userMessage];
    onUpdateSessionMessages(messagesWithUser);

    // Step 2: Upload imagery to Firebase Storage, track storage paths
    let attachmentMeta = []; // [{ name, storagePath, localUrl }]
    for (const file of filesToUpload) {
      if (user && !isDemoMode) {
        try {
          const storagePath = `users/${user.id}/imagery/${Date.now()}_${file.name}`;
          const storageRef = ref(storage, storagePath);
          await uploadBytes(storageRef, file);
          await addDoc(collection(db, 'users', user.id, 'imagery'), {
            name: file.name,
            storagePath,
            uploadedAt: Date.now(),
            size: file.size
          });
          attachmentMeta.push({ name: file.name, storagePath, localUrl: URL.createObjectURL(file) });
        } catch (uploadErr) {
          console.error('Storage upload error:', uploadErr);
          attachmentMeta.push({ name: file.name, storagePath: null, localUrl: URL.createObjectURL(file) });
        }
      } else {
        attachmentMeta.push({ name: file.name, storagePath: null, localUrl: URL.createObjectURL(file) });
      }
    }

    const userMessageWithAttachments = {
      ...userMessage,
      attachments: attachmentMeta.map(m => ({ name: m.name, url: m.localUrl }))
    };
    const updatedMessages = messagesWithUser.map(m =>
      m.id === userMessage.id ? userMessageWithAttachments : m
    );
    onUpdateSessionMessages(updatedMessages);

    try {
      const API_BASE = '/api';

      // Get Firebase ID token for backend auth
      let idToken = null;
      if (user && !isDemoMode && auth.currentUser) {
        idToken = await auth.currentUser.getIdToken();
      }

      // Use /api/analyze with the storage path (preferred: first uploaded image)
      const uploadedFile = attachmentMeta.find(m => m.storagePath);
      if (idToken && uploadedFile?.storagePath) {
        const response = await axios.post(
          `${API_BASE}/analyze`,
          {
            image_path: uploadedFile.storagePath,
            query: textToSend,
            max_new_tokens: 256,
          },
          {
            headers: { Authorization: `Bearer ${idToken}` },
            timeout: 120000,
          }
        );
        const botMessage = {
          id: (Date.now() + 1).toString(),
          sender: 'assistant',
          text: response.data.answer,
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        };
        onUpdateSessionMessages([...updatedMessages, botMessage]);
      } else {
        // Text-only or demo mode: show friendly message
        const botMessage = {
          id: (Date.now() + 1).toString(),
          sender: 'assistant',
          text: isDemoMode
            ? '⚠️ Sign in to analyse satellite imagery with Qwen AI.'
            : '⚠️ Please attach a satellite image to run analysis.',
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        };
        onUpdateSessionMessages([...updatedMessages, botMessage]);
      }

    } catch (error) {
      console.error('Backend error:', error?.response?.status, error?.message);
      const botError = {
        id: (Date.now() + 1).toString(),
        sender: 'assistant',
        isError: true,
        text: `Analysis failed (${error?.response?.status || 'error'}). Please try again.`,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      onUpdateSessionMessages([...updatedMessages, botError]);
    } finally {
      setIsProcessing(false);
    }
  };

  const samplePrompts = [
    { title: "Urban Infrastructure", text: "Identify new building constructions, road expansions, and urban density changes." },
    { title: "Land Cover & Canopy", text: "Classify vegetation canopy, water bodies, agricultural parcels, and bare ground." },
    { title: "Temporal Progression", text: "Analyze historical satellite sequence to map environmental and morphological shifts." }
  ];

  const messages = activeSession?.messages || [];

  return (
    <div style={{
      flex: 1,
      display: 'flex',
      flexDirection: 'column',
      height: '100vh',
      backgroundColor: 'var(--bg-primary)',
      position: 'relative'
    }}>
      {/* Header Bar */}
      <div style={{
        padding: '0.9rem 1.75rem',
        borderBottom: '1px solid var(--border-subtle)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        background: 'var(--bg-secondary)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <h2 style={{ fontSize: '0.95rem', fontWeight: 600, color: 'var(--text-primary)', letterSpacing: '-0.01em' }}>
            {activeSession?.title || 'New Analysis Session'}
          </h2>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        </div>
      </div>

      {/* Messages Feed */}
      <div style={{
        flex: 1,
        overflowY: 'auto',
        padding: '2rem',
        display: 'flex',
        flexDirection: 'column',
        gap: '1.5rem'
      }}>
        {messages.length === 0 ? (
          /* Clean Minimal Empty State */
          <div style={{
            margin: 'auto',
            maxWidth: '680px',
            width: '100%',
            textAlign: 'center',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: '1.25rem',
            padding: '2rem 1rem'
          }}>
            <img
              src={logoSrc}
              alt="AKASHA"
              style={{
                width: '56px',
                height: '56px',
                borderRadius: '12px',
                objectFit: 'contain',
                marginBottom: '4px'
              }}
              draggable="false"
            />

            <div>
              <h3 style={{ fontSize: '1.45rem', fontWeight: 700, letterSpacing: '-0.02em', color: 'var(--text-primary)', marginBottom: '6px' }}>
                Earth Observation Intelligence
              </h3>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', lineHeight: 1.5, maxWidth: '520px', margin: '0 auto' }}>
                Upload satellite imagery and ask queries. AKASHA routes your request to specialized models like GeoChat, TEOChat, and M2CD.
              </p>
            </div>

            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
              gap: '12px',
              width: '100%',
              marginTop: '0.5rem'
            }}>
              {samplePrompts.map((p, idx) => (
                <div
                  key={idx}
                  onClick={() => handleSend(p.text)}
                  className="glass-interactive"
                  style={{
                    padding: '14px 16px',
                    borderRadius: '10px',
                    textAlign: 'left',
                    cursor: 'pointer',
                    fontSize: '0.84rem'
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '6px' }}>
                    <strong style={{ color: 'var(--text-primary)', fontSize: '0.86rem' }}>
                      {p.title}
                    </strong>
                    <ArrowUpRight size={14} color="var(--text-muted)" />
                  </div>
                  <span style={{ color: 'var(--text-muted)', fontSize: '0.78rem', lineHeight: 1.4, display: 'block' }}>
                    {p.text}
                  </span>
                </div>
              ))}
            </div>
          </div>
        ) : (
          messages.map(msg => {
            const isUser = msg.sender === 'user';
            return (
              <div
                key={msg.id}
                style={{
                  display: 'flex',
                  gap: '12px',
                  alignSelf: isUser ? 'flex-end' : 'flex-start',
                  maxWidth: isUser ? '72%' : '84%',
                  animation: 'fadeIn 0.25s ease-out'
                }}
              >
                {!isUser && (
                  <div style={{
                    width: '32px',
                    height: '32px',
                    borderRadius: '8px',
                    background: 'var(--bg-card)',
                    border: '1px solid var(--border-medium)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    flexShrink: 0
                  }}>
                    <Bot size={17} color="var(--text-primary)" />
                  </div>
                )}

                <div style={{
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '6px',
                  width: '100%'
                }}>
                  <div style={{
                    padding: '12px 16px',
                    borderRadius: isUser ? '14px 14px 2px 14px' : '14px 14px 14px 2px',
                    background: isUser ? 'var(--bg-card)' : 'var(--bg-secondary)',
                    border: isUser ? '1px solid var(--border-medium)' : '1px solid var(--border-subtle)',
                    color: 'var(--text-primary)',
                    boxShadow: '0 2px 8px rgba(0,0,0,0.06)'
                  }}>
                    {/* Attachments inside user message */}
                    {msg.attachments && msg.attachments.length > 0 && (
                      <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', marginBottom: '8px' }}>
                        {msg.attachments.map((att, i) => {
                          const imgUrl = typeof att === 'string' ? att : (att?.url || att?.preview);
                          const imgName = typeof att === 'string' ? 'Satellite Image' : (att?.name || 'Satellite Image');
                          return (
                            <div key={i} style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                              {imgUrl ? (
                                <img
                                  src={imgUrl}
                                  alt={imgName}
                                  onClick={() => setPreviewImage({ url: imgUrl, name: imgName })}
                                  title="Click to enlarge"
                                  style={{
                                    maxWidth: '220px',
                                    maxHeight: '150px',
                                    borderRadius: '8px',
                                    objectFit: 'cover',
                                    border: '1px solid var(--border-subtle)',
                                    cursor: 'pointer',
                                    transition: 'border-color 0.15s ease'
                                  }}
                                  onMouseEnter={e => e.currentTarget.style.borderColor = 'var(--border-strong)'}
                                  onMouseLeave={e => e.currentTarget.style.borderColor = 'var(--border-subtle)'}
                                />
                              ) : (
                                <span style={{
                                  display: 'inline-flex',
                                  alignItems: 'center',
                                  gap: '4px',
                                  background: 'var(--bg-card-hover)',
                                  border: '1px solid var(--border-subtle)',
                                  padding: '4px 8px',
                                  borderRadius: '6px',
                                  fontSize: '0.74rem',
                                  color: 'var(--text-primary)'
                                }}>
                                  <ImageIcon size={12} /> {imgName}
                                </span>
                              )}
                            </div>
                          );
                        })}
                      </div>
                    )}

                    <p style={{ 
                      fontSize: '0.92rem', 
                      lineHeight: 1.55, 
                      whiteSpace: 'pre-wrap', 
                      color: msg.isError ? '#ef4444' : 'var(--text-primary)' 
                    }}>
                      {msg.text}
                    </p>
                  </div>

                  {/* Message Timestamp */}
                  {msg.timestamp && (
                    <div style={{
                      fontSize: '0.72rem',
                      color: 'var(--text-muted)',
                      textAlign: isUser ? 'right' : 'left',
                      paddingLeft: isUser ? '0' : '2px',
                      paddingRight: isUser ? '2px' : '0'
                    }}>
                      <span>{msg.timestamp}</span>
                    </div>
                  )}
                </div>

                {isUser && (
                  <div style={{
                    width: '32px',
                    height: '32px',
                    borderRadius: '8px',
                    background: 'var(--bg-card)',
                    border: '1px solid var(--border-subtle)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    flexShrink: 0
                  }}>
                    <User size={16} color="var(--text-secondary)" />
                  </div>
                )}
              </div>
            );
          })
        )}

        {isProcessing && (
          <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
            <div style={{
              width: '32px',
              height: '32px',
              borderRadius: '8px',
              background: 'var(--bg-card)',
              border: '1px solid var(--border-subtle)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}>
              <Bot size={17} color="var(--text-primary)" />
            </div>
            <div style={{
              padding: '10px 16px',
              borderRadius: '12px',
              background: 'var(--bg-card)',
              border: '1px solid var(--border-subtle)',
              display: 'flex',
              alignItems: 'center',
              gap: '10px'
            }}>
              <div className="spinner" />
              <span style={{ fontSize: '0.86rem', color: 'var(--text-secondary)' }}>Processing imagery with specialist model...</span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div style={{
        padding: '1rem 1.75rem 1.25rem 1.75rem',
        background: 'var(--bg-secondary)',
        borderTop: '1px solid var(--border-subtle)'
      }}>
        {/* Attachment Chips */}
        {selectedFiles && selectedFiles.length > 0 && (
          <div style={{
            display: 'flex',
            gap: '6px',
            overflowX: 'auto',
            paddingBottom: '8px',
            marginBottom: '4px'
          }}>
            {selectedFiles.map((file, idx) => (
              <div key={idx} style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '6px',
                background: 'var(--bg-card)',
                border: '1px solid var(--border-medium)',
                borderRadius: '6px',
                padding: '3px 8px',
                fontSize: '0.76rem',
                color: 'var(--text-primary)'
              }}>
                <ImageIcon size={13} color="var(--text-secondary)" />
                <span>{file.name}</span>
                <X
                  size={13}
                  style={{ cursor: 'pointer', marginLeft: '4px', color: 'var(--text-muted)' }}
                  onClick={() => onFileSelect(selectedFiles.filter((_, i) => i !== idx))}
                />
              </div>
            ))}
          </div>
        )}

        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          {/* File Attachment Button */}
          <label 
            className="btn-secondary" 
            style={{ padding: '10px', borderRadius: '8px', cursor: 'pointer' }}
            title="Attach Satellite Imagery"
          >
            <input
              type="file"
              multiple
              accept=".tiff,.tif,.png,.jpeg,.jpg"
              style={{ display: 'none' }}
              onChange={(e) => {
                if (e.target.files) {
                  onFileSelect([...selectedFiles, ...Array.from(e.target.files)]);
                }
              }}
            />
            <Paperclip size={17} />
          </label>

          {/* Text Input */}
          <input
            type="text"
            className="glass-input"
            placeholder="Ask about imagery features, segmentation, or morphological changes..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
            disabled={isProcessing}
            style={{ borderRadius: '8px', padding: '11px 14px' }}
          />

          {/* Send Button */}
          <button
            className="btn-primary"
            onClick={() => handleSend()}
            disabled={((!query.trim() && (!selectedFiles || selectedFiles.length === 0)) || isProcessing)}
            style={{ 
              padding: '10px 18px', 
              borderRadius: '8px'
            }}
          >
            {isProcessing ? <div className="spinner" /> : <Send size={16} />}
            <span>Run</span>
          </button>
        </div>
      </div>

      {/* Full-screen Lightbox Modal */}
      {previewImage && (
        <div 
          onClick={() => setPreviewImage(null)}
          style={{
            position: 'fixed',
            inset: 0,
            zIndex: 1000,
            background: 'rgba(0, 0, 0, 0.88)',
            backdropFilter: 'blur(8px)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '2rem',
            animation: 'fadeIn 0.2s ease-out'
          }}
        >
          <div 
            onClick={e => e.stopPropagation()}
            style={{
              position: 'relative',
              maxWidth: '90vw',
              maxHeight: '90vh',
              display: 'flex',
              flexDirection: 'column',
              background: 'var(--bg-card)',
              border: '1px solid var(--border-medium)',
              borderRadius: '12px',
              padding: '1.25rem',
              boxShadow: '0 24px 60px rgba(0,0,0,0.5)'
            }}
          >
            <div style={{
              display: 'flex',
              width: '100%',
              alignItems: 'center',
              justifyContent: 'space-between',
              marginBottom: '0.85rem',
              gap: '12px'
            }}>
              <h4 style={{ color: 'var(--text-primary)', fontSize: '0.92rem', fontWeight: 600 }}>{previewImage.name || 'Satellite Imagery'}</h4>
              <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                <a
                  href={previewImage.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="btn-secondary"
                  style={{ fontSize: '0.76rem', padding: '5px 10px' }}
                >
                  Open Original
                </a>
                <button
                  onClick={() => setPreviewImage(null)}
                  style={{ background: 'transparent', border: 'none', color: 'var(--text-primary)', cursor: 'pointer', padding: '4px', display: 'flex' }}
                >
                  <X size={18} />
                </button>
              </div>
            </div>
            <img
              src={previewImage.url}
              alt={previewImage.name || 'Full preview'}
              style={{
                maxWidth: '100%',
                maxHeight: '75vh',
                borderRadius: '8px',
                objectFit: 'contain',
                border: '1px solid var(--border-subtle)'
              }}
            />
          </div>
        </div>
      )}
    </div>
  );
};

export default ChatInterface;
