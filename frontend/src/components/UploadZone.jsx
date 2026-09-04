import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { UploadCloud, Image as ImageIcon, X, Plus } from 'lucide-react';

const UploadZone = ({ onFileSelect, selectedFiles }) => {
  const onDrop = useCallback(acceptedFiles => {
    if (acceptedFiles && acceptedFiles.length > 0) {
      onFileSelect([...selectedFiles, ...acceptedFiles]);
    }
  }, [onFileSelect, selectedFiles]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ 
    onDrop,
    accept: {
      'image/tiff': ['.tiff', '.tif'],
      'image/png': ['.png'],
      'image/jpeg': ['.jpeg', '.jpg']
    }
  });

  return (
    <div className="glass" style={{ display: 'flex', flexDirection: 'column', gap: '1rem', padding: '1.25rem' }}>
      <h2 style={{ fontSize: '1.05rem', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-primary)' }}>
        <UploadCloud size={18} color="var(--text-primary)" /> Upload Imagery
      </h2>
      
      {selectedFiles.length === 0 ? (
        <div 
          {...getRootProps()} 
          style={{
            border: `1px dashed ${isDragActive ? 'var(--accent-primary)' : 'var(--border-medium)'}`,
            borderRadius: '10px',
            padding: '2rem',
            textAlign: 'center',
            cursor: 'pointer',
            transition: 'all 0.18s ease',
            backgroundColor: isDragActive ? 'var(--glass-bg-hover)' : 'transparent'
          }}
        >
          <input {...getInputProps()} />
          <UploadCloud size={36} color="var(--text-muted)" style={{ margin: '0 auto 0.75rem' }} />
          <p style={{ marginBottom: '0.25rem', fontSize: '0.88rem', color: 'var(--text-primary)', fontWeight: 500 }}>
            {isDragActive ? "Drop imagery files here..." : "Drag and drop satellite imagery here"}
          </p>
          <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
            or click to select from your device
          </p>
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          {selectedFiles.map((file, idx) => (
            <div key={idx} style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              background: 'var(--bg-secondary)',
              padding: '0.75rem 1rem',
              borderRadius: '8px',
              border: '1px solid var(--border-subtle)'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px', overflow: 'hidden' }}>
                <ImageIcon size={20} color="var(--text-secondary)" />
                <div style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  <p style={{ fontWeight: 500, fontSize: '0.86rem', color: 'var(--text-primary)' }}>{file.name}</p>
                  <p style={{ fontSize: '0.74rem', color: 'var(--text-muted)' }}>
                    {(file.size / (1024 * 1024)).toFixed(2)} MB
                  </p>
                </div>
              </div>
              <button 
                onClick={(e) => {
                  e.stopPropagation();
                  onFileSelect(selectedFiles.filter((_, i) => i !== idx));
                }}
                style={{
                  background: 'transparent',
                  border: 'none',
                  cursor: 'pointer',
                  color: 'var(--text-muted)',
                  padding: '4px',
                  display: 'flex',
                  alignItems: 'center',
                  borderRadius: '4px'
                }}
                onMouseEnter={e => e.currentTarget.style.color = '#ef4444'}
                onMouseLeave={e => e.currentTarget.style.color = 'var(--text-muted)'}
              >
                <X size={16} />
              </button>
            </div>
          ))}
          
          <div 
            {...getRootProps()} 
            style={{
              border: `1px dashed var(--border-subtle)`,
              borderRadius: '8px',
              padding: '0.75rem',
              textAlign: 'center',
              cursor: 'pointer',
              transition: 'all 0.18s ease',
              backgroundColor: isDragActive ? 'var(--glass-bg-hover)' : 'transparent',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '6px',
              color: 'var(--text-secondary)',
              fontSize: '0.82rem'
            }}
          >
            <input {...getInputProps()} />
            <Plus size={16} color="var(--text-secondary)" />
            <span>Add more imagery</span>
          </div>
        </div>
      )}
      
      <div style={{ fontSize: '0.74rem', color: 'var(--text-muted)' }}>
        Supported formats: GeoTIFF, TIFF, PNG, JPEG
      </div>
    </div>
  );
};

export default UploadZone;
