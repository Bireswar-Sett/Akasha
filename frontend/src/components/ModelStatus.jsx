import React from 'react';
import { CheckCircle, Database } from 'lucide-react';

const ModelStatus = () => {
  const models = [
    { name: 'GeoChat', status: 'Available', color: 'var(--accent-secondary)' },
    { name: 'TEOChat', status: 'Available', color: 'var(--accent-primary)' },
    { name: 'GeoVision', status: 'Available', color: '#4ade80' },
    { name: 'M2CD', status: 'Available', color: '#f472b6' },
  ];

  return (
    <div className="glass">
      <h2 style={{ fontSize: '1.2rem', fontWeight: 600, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
        <Database size={20} color="var(--accent-primary)" /> Connected Models
      </h2>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
        {models.map(model => (
          <div key={model.name} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <div style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: model.color }} />
              <span style={{ fontWeight: 500 }}>{model.name}</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '4px', color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
              <CheckCircle size={14} color="#4ade80" />
              {model.status}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ModelStatus;
