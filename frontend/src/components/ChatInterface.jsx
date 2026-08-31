import React, { useState, useRef, useEffect } from 'react';
import { 
  Send, 
  Paperclip, 
  X, 
  Sparkles, 
  Bot, 
  User, 
  Layers, 
  TrendingUp, 
  Activity, 
  CheckCircle, 
  Image as ImageIcon,
  Compass
} from 'lucide-react';
import axios from 'axios';

const ChatInterface = ({ 
  selectedFiles, 
  onFileSelect, 
  activeSession, 
  onUpdateSessionMessages 
}) => {
  const [query, setQuery] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [activeSession?.messages, isProcessing]);

  const handleSend = async (overrideQuery = null) => {
    const textToSend = overrideQuery || query;
    if (!textToSend.trim() || isProcessing) return;

    const userMessage = {
      id: Date.now().toString(),
      sender: 'user',
      text: textToSend,
      attachments: [...selectedFiles],
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    const updatedMessages = [...(activeSession?.messages || []), userMessage];
    onUpdateSessionMessages(updatedMessages);
    setQuery('');
    setIsProcessing(true);

    try {
      // 1. Orchestration
      const orchestrateData = new FormData();
      orchestrateData.append('query', textToSend);
      if (selectedFiles && selectedFiles.length > 0) {
        selectedFiles.forEach(file => orchestrateData.append('images', file));
      }

      const orchestrateResponse = await axios.post('http://127.0.0.1:8000/api/orchestrate', orchestrateData);
      const selectedModel = orchestrateResponse.data.selected_model;
      const confidence = orchestrateResponse.data.confidence;

      // 2. Execution
      const executeData = new FormData();
      executeData.append('model_name', selectedModel);
      executeData.append('query', textToSend);
      if (selectedFiles && selectedFiles.length > 0) {
        selectedFiles.forEach(file => executeData.append('images', file));
      }

      const executeResponse = await axios.post('http://127.0.0.1:8000/api/execute', executeData);

      const botMessage = {
        id: (Date.now() + 1).toString(),
        sender: 'assistant',
        text: executeResponse.data.response,
        modelUsed: selectedModel,
        confidence: confidence,
        visualEvidence: executeResponse.data.visual_evidence_url,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };

      onUpdateSessionMessages([...updatedMessages, botMessage]);

    } catch (error) {
      console.error('Error processing satellite request:', error);
      const errorMessage = {
        id: (Date.now() + 1).toString(),
        sender: 'assistant',
        isError: true,
        text: "Could not complete analysis. Please ensure the Python backend API (port 8000) is active.",
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      onUpdateSessionMessages([...updatedMessages, errorMessage]);
    } finally {
      setIsProcessing(false);
    }
  };

  const samplePrompts = [
    { title: "Detect Urban Growth", text: "Identify new building constructions and urban development changes." },
    { title: "Land Cover Breakdown", text: "Classify vegetation, water bodies, and developed areas." },
    { title: "Temporal Comparison", text: "Analyze historical imagery sequence for temporal environmental progression." }
  ];

  const messages = activeSession?.messages || [];

  return (
    <div style={{
      flex: 1,
      display: 'flex',
      flexDirection: 'column',
      height: '100vh',
      backgroundColor: 'transparent',
      position: 'relative'
    }}>
      {/* Header Bar */}
      <div style={{
        padding: '1rem 2rem',
        borderBottom: '1px solid var(--glass-border)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        background: 'rgba(10, 10, 26, 0.4)',
        backdropFilter: 'blur(10px)'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <h2 style={{ fontSize: '1.1rem', fontWeight: 600, color: '#fff' }}>
            {activeSession?.title || 'New Satellite Intelligence Analysis'}
          </h2>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: '#4EFA7B' }}></span>
            Orchestrator Ready
          </span>
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
          /* Welcome Empty State */
          <div style={{
            margin: 'auto',
            maxWidth: '650px',
            textAlign: 'center',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: '1.2rem',
            padding: '2rem'
          }}>
            <div style={{
              width: '64px',
              height: '64px',
              borderRadius: '20px',
              background: 'linear-gradient(135deg, rgba(108, 99, 255, 0.2), rgba(0, 212, 255, 0.2))',
              border: '1px solid var(--glass-border)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}>
              <Compass size={32} color="var(--accent-secondary)" />
            </div>
            <h3 style={{ fontSize: '1.6rem', fontWeight: 600 }}>What would you like to analyze?</h3>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.95rem', lineHeight: 1.5 }}>
              Upload satellite imagery and ask queries. AKASHA automatically routes your prompt to specialist models like GeoChat, TEOChat, SkySense, or M2CD.
            </p>

            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
              gap: '12px',
              width: '100%',
              marginTop: '1rem'
            }}>
              {samplePrompts.map((p, idx) => (
                <div
                  key={idx}
                  onClick={() => handleSend(p.text)}
                  className="glass"
                  style={{
                    padding: '14px',
                    borderRadius: '12px',
                    textAlign: 'left',
                    cursor: 'pointer',
                    fontSize: '0.85rem'
                  }}
                >
                  <strong style={{ display: 'block', color: 'var(--accent-secondary)', marginBottom: '4px' }}>
                    {p.title}
                  </strong>
                  <span style={{ color: 'var(--text-secondary)', fontSize: '0.78rem' }}>
                    {p.text}
                  </span>
                </div>
              ))}
            </div>
          </div>
        ) : (
          messages.map(msg => (
            <div
              key={msg.id}
              style={{
                display: 'flex',
                gap: '12px',
                alignSelf: msg.sender === 'user' ? 'flex-end' : 'flex-start',
                maxWidth: msg.sender === 'user' ? '70%' : '85%',
                animation: 'fadeIn 0.3s ease-out'
              }}
            >
              {msg.sender === 'assistant' && (
                <div style={{
                  width: '36px',
                  height: '36px',
                  borderRadius: '10px',
                  background: 'linear-gradient(135deg, var(--accent-primary), var(--accent-secondary))',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  flexShrink: 0
                }}>
                  <Bot size={20} color="#fff" />
                </div>
              )}

              <div style={{
                display: 'flex',
                flexDirection: 'column',
                gap: '8px'
              }}>
                <div style={{
                  padding: '14px 18px',
                  borderRadius: msg.sender === 'user' ? '18px 18px 2px 18px' : '18px 18px 18px 2px',
                  background: msg.sender === 'user' 
                    ? 'linear-gradient(135deg, var(--accent-primary), #4834d4)' 
                    : 'rgba(255, 255, 255, 0.05)',
                  border: msg.sender === 'user' ? 'none' : '1px solid var(--glass-border)',
                  color: '#fff',
                  boxShadow: '0 4px 15px rgba(0,0,0,0.1)'
                }}>
                  {/* Attachments inside user message */}
                  {msg.attachments && msg.attachments.length > 0 && (
                    <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap', marginBottom: '8px' }}>
                      {msg.attachments.map((file, i) => (
                        <span key={i} style={{
                          display: 'inline-flex',
                          alignItems: 'center',
                          gap: '4px',
                          background: 'rgba(0,0,0,0.2)',
                          padding: '4px 8px',
                          borderRadius: '6px',
                          fontSize: '0.75rem'
                        }}>
                          <ImageIcon size={12} /> {file.name}
                        </span>
                      ))}
                    </div>
                  )}

                  <p style={{ fontSize: '0.95rem', lineHeight: 1.5, whitespace: 'pre-wrap' }}>
                    {msg.text}
                  </p>
                </div>

                {/* Assistant Specialist Metadata Badge */}
                {msg.sender === 'assistant' && !msg.isError && (
                  <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '12px',
                    fontSize: '0.78rem',
                    color: 'var(--text-secondary)',
                    paddingLeft: '4px'
                  }}>
                    <span style={{
                      background: 'rgba(0, 212, 255, 0.15)',
                      color: 'var(--accent-secondary)',
                      padding: '2px 8px',
                      borderRadius: '12px',
                      border: '1px solid rgba(0, 212, 255, 0.3)',
                      fontWeight: 500
                    }}>
                      Model: {msg.modelUsed}
                    </span>
                    <span>Confidence: {msg.confidence}%</span>
                    <span>{msg.timestamp}</span>
                  </div>
                )}
              </div>

              {msg.sender === 'user' && (
                <div style={{
                  width: '36px',
                  height: '36px',
                  borderRadius: '10px',
                  background: 'rgba(255, 255, 255, 0.1)',
                  border: '1px solid var(--glass-border)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  flexShrink: 0
                }}>
                  <User size={18} color="var(--text-secondary)" />
                </div>
              )}
            </div>
          ))
        )}

        {isProcessing && (
          <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
            <div style={{
              width: '36px',
              height: '36px',
              borderRadius: '10px',
              background: 'linear-gradient(135deg, var(--accent-primary), var(--accent-secondary))',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}>
              <Bot size={20} color="#fff" />
            </div>
            <div className="glass" style={{ padding: '12px 18px', borderRadius: '18px 18px 18px 2px', display: 'flex', alignItems: 'center', gap: '10px' }}>
              <div className="spinner" />
              <span style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>Orchestrating model analysis...</span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div style={{
        padding: '1.2rem 2rem',
        background: 'rgba(10, 10, 26, 0.6)',
        backdropFilter: 'blur(12px)',
        borderTop: '1px solid var(--glass-border)'
      }}>
        {/* Attachment Chips standard preview before sending */}
        {selectedFiles && selectedFiles.length > 0 && (
          <div style={{
            display: 'flex',
            gap: '8px',
            overflowX: 'auto',
            paddingBottom: '8px',
            marginBottom: '6px'
          }}>
            {selectedFiles.map((file, idx) => (
              <div key={idx} style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '6px',
                background: 'rgba(108, 99, 255, 0.15)',
                border: '1px solid rgba(108, 99, 255, 0.3)',
                borderRadius: '8px',
                padding: '4px 10px',
                fontSize: '0.78rem',
                color: '#fff'
              }}>
                <ImageIcon size={14} color="var(--accent-secondary)" />
                <span>{file.name}</span>
                <X
                  size={14}
                  style={{ cursor: 'pointer', marginLeft: '4px' }}
                  onClick={() => onFileSelect(selectedFiles.filter((_, i) => i !== idx))}
                />
              </div>
            ))}
          </div>
        )}

        <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
          {/* Paperclip file uploader input */}
          <label 
            className="glass-button" 
            style={{ padding: '12px', borderRadius: '12px', cursor: 'pointer' }}
            title="Attach Imagery"
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
            <Paperclip size={18} color="var(--accent-secondary)" />
          </label>

          <input
            type="text"
            className="glass-input"
            placeholder="Ask AKASHA about satellite imagery features, land cover, or changes..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
            disabled={isProcessing}
            style={{ borderRadius: '14px', padding: '14px 18px' }}
          />

          <button
            className="glass-button"
            onClick={() => handleSend()}
            disabled={!query.trim() || isProcessing}
            style={{ padding: '12px 20px', borderRadius: '12px' }}
          >
            {isProcessing ? <div className="spinner" /> : <Send size={18} />}
            <span>Analyze</span>
          </button>
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;
