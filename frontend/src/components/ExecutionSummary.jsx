import React from 'react';
import { Layers, Activity, FileText, CheckCircle2, AlertTriangle } from 'lucide-react';

const ExecutionSummary = ({ result, isProcessing }) => {
  if (isProcessing) {
    return (
      <div className="glass" style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '160px', padding: '1.5rem' }}>
        <div style={{ textAlign: 'center' }}>
          <div className="spinner" style={{ margin: '0 auto 0.75rem', width: '24px', height: '24px' }} />
          <p style={{ fontSize: '0.86rem', color: 'var(--text-secondary)' }}>Processing geospatial inference...</p>
        </div>
      </div>
    );
  }

  if (!result) return null;

  if (result.error) {
    return (
      <div className="glass" style={{ border: '1px solid rgba(239, 68, 68, 0.3)', padding: '1.25rem' }}>
        <p style={{ color: '#ef4444', display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.88rem' }}>
          <AlertTriangle size={16} /> {result.error}
        </p>
      </div>
    );
  }

  return (
    <div className="glass animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem', padding: '1.25rem' }}>
      <h3 style={{ fontSize: '1.05rem', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-primary)' }}>
        <FileText size={18} color="var(--text-primary)" /> Analysis Summary
      </h3>
      
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.85rem' }}>
        <div>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.76rem', marginBottom: '3px', textTransform: 'uppercase' }}>Task Type</p>
          <p style={{ fontWeight: 500, fontSize: '0.88rem', display: 'flex', alignItems: 'center', gap: '6px', color: 'var(--text-primary)' }}>
            <Layers size={15} color="var(--text-secondary)" /> {result.task}
          </p>
        </div>
        <div>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.76rem', marginBottom: '3px', textTransform: 'uppercase' }}>Selected Specialist</p>
          <p style={{ fontWeight: 600, fontSize: '0.88rem', color: 'var(--text-primary)' }}>{result.model}</p>
        </div>
        <div>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.76rem', marginBottom: '3px', textTransform: 'uppercase' }}>Confidence</p>
          <p style={{ fontWeight: 500, fontSize: '0.88rem', display: 'flex', alignItems: 'center', gap: '6px', color: 'var(--text-primary)' }}>
            <Activity size={15} color="var(--text-primary)" /> {result.confidence}%
          </p>
        </div>
      </div>

      <div>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.76rem', marginBottom: '6px', textTransform: 'uppercase' }}>Visual Evidence</p>
        <div style={{ 
          width: '100%', 
          height: '140px', 
          background: 'var(--bg-secondary)', 
          borderRadius: '8px',
          border: '1px solid var(--border-subtle)',
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          color: 'var(--text-muted)',
          fontSize: '0.82rem'
        }}>
          Inference Segmentation Map
        </div>
      </div>

      <div style={{ background: 'var(--bg-secondary)', padding: '0.85rem 1rem', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
        <p style={{ display: 'flex', alignItems: 'flex-start', gap: '8px', fontSize: '0.86rem', color: 'var(--text-primary)', lineHeight: 1.5 }}>
          <CheckCircle2 size={16} color="var(--text-primary)" style={{ flexShrink: 0, marginTop: '2px' }} />
          <span><strong style={{ color: 'var(--text-primary)' }}>Output:</strong> {result.response}</span>
        </p>
      </div>
    </div>
  );
};

export default ExecutionSummary;
