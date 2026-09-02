import React from 'react';
import { Database } from 'lucide-react';

const ModelStatus = () => {
  const models = [
    { name: 'GeoChat', status: 'Available', type: 'High-Res VLM' },
    { name: 'TEOChat', status: 'Available', type: 'Temporal Earth Observation' },
    { name: 'GeoVision', status: 'Available', type: 'Multi-Spectral Segmentation' },
    { name: 'M2CD', status: 'Available', type: 'Change Detection' },
  ];

  return (
    <div className="glass" style={{ padding: '1.25rem' }}>
      <h2 style={{ fontSize: '1.05rem', fontWeight: 600, marginBottom: '0.85rem', display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-primary)' }}>
        <Database size={18} color="var(--text-primary)" /> Specialist Neural Models
      </h2>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.65rem' }}>
        {models.map(model => (
          <div 
            key={model.name} 
            style={{ 
              display: 'flex', 
              justifyContent: 'space-between', 
              alignItems: 'center',
              padding: '6px 8px',
              borderRadius: '6px',
              background: 'var(--bg-secondary)',
              border: '1px solid var(--border-subtle)'
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <div style={{ width: '6px', height: '6px', borderRadius: '50%', backgroundColor: 'var(--text-primary)' }} />
              <span style={{ fontWeight: 500, fontSize: '0.84rem', color: 'var(--text-primary)' }}>{model.name}</span>
              <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>{model.type}</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '4px', color: 'var(--text-secondary)', fontSize: '0.76rem' }}>
              {model.status}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ModelStatus;
