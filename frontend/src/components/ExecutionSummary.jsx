import React from 'react';
import { Layers, Activity, FileText, CheckCircle } from 'lucide-react';

const ExecutionSummary = ({ result, isProcessing }) => {
  if (isProcessing) {
    return (
      <div className="glass animate-pulse" style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '200px' }}>
        <div style={{ textAlign: 'center' }}>
          <div className="spinner" style={{ margin: '0 auto 1rem', width: '32px', height: '32px' }} />
          <p>Analyzing with specialist AI...</p>
        </div>
      </div>
    );
  }

  if (!result) return null;

  if (result.error) {
    return (
      <div className="glass" style={{ border: '1px solid rgba(255, 99, 132, 0.5)' }}>
        <p style={{ color: '#ff6384' }}>{result.error}</p>
      </div>
    );
  }

  return (
    <div className="glass animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      <h3 style={{ fontSize: '1.2rem', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '8px' }}>
        <FileText size={20} color="var(--accent-primary)" /> Execution Summary
      </h3>
      
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
        <div>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginBottom: '4px' }}>Task</p>
          <p style={{ fontWeight: 500, display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Layers size={16} /> {result.task}
          </p>
        </div>
        <div>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginBottom: '4px' }}>Model</p>
          <p style={{ fontWeight: 500, color: 'var(--accent-secondary)' }}>{result.model}</p>
        </div>
        <div>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginBottom: '4px' }}>Confidence</p>
          <p style={{ fontWeight: 500, display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Activity size={16} color="#4ade80" /> {result.confidence}%
          </p>
        </div>
      </div>

      <div>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginBottom: '8px' }}>Visual Evidence</p>
        <div style={{ 
          width: '100%', 
          height: '150px', 
          background: 'rgba(0,0,0,0.2)', 
          borderRadius: '8px',
          border: '1px solid var(--glass-border)',
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          color: 'var(--text-secondary)'
        }}>
          [Visual Evidence Placeholder - Box/Heatmap]
        </div>
      </div>

      <div style={{ background: 'rgba(74, 222, 128, 0.1)', padding: '1rem', borderRadius: '8px', border: '1px solid rgba(74, 222, 128, 0.2)' }}>
        <p style={{ display: 'flex', alignItems: 'flex-start', gap: '8px' }}>
          <CheckCircle size={20} color="#4ade80" style={{ flexShrink: 0, marginTop: '2px' }} />
          <span><strong style={{ color: '#4ade80' }}>Response:</strong> {result.response}</span>
        </p>
      </div>
    </div>
  );
};

export default ExecutionSummary;
