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
    <div className="glass" style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      <h2 style={{ fontSize: '1.2rem', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '8px' }}>
        <UploadCloud size={20} color="var(--accent-primary)" /> Upload Imagery
      </h2>
      
      {selectedFiles.length === 0 ? (
        <div 
          {...getRootProps()} 
          style={{
            border: `2px dashed ${isDragActive ? 'var(--accent-primary)' : 'var(--glass-border)'}`,
            borderRadius: '12px',
            padding: '2rem',
            textAlign: 'center',
            cursor: 'pointer',
            transition: 'all 0.3s ease',
            backgroundColor: isDragActive ? 'rgba(108, 99, 255, 0.1)' : 'transparent'
          }}
        >
          <input {...getInputProps()} />
          <UploadCloud size={48} color="var(--text-secondary)" style={{ margin: '0 auto 1rem' }} />
          <p style={{ marginBottom: '0.5rem' }}>
            {isDragActive ? "Drop the image here..." : "Drag 'n' drop satellite images here"}
          </p>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
            or click to select files
          </p>
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          {selectedFiles.map((file, idx) => (
            <div key={idx} style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              background: 'rgba(255,255,255,0.05)',
              padding: '1rem',
              borderRadius: '12px',
              border: '1px solid var(--glass-border)'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px', overflow: 'hidden' }}>
                <ImageIcon size={24} color="var(--accent-secondary)" />
                <div style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  <p style={{ fontWeight: 500, fontSize: '0.95rem' }}>{file.name}</p>
                  <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
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
                  color: 'var(--text-secondary)',
                  padding: '4px'
                }}
              >
                <X size={20} />
              </button>
            </div>
          ))}
          
          <div 
            {...getRootProps()} 
            style={{
              border: `2px dashed ${isDragActive ? 'var(--accent-primary)' : 'var(--glass-border)'}`,
              borderRadius: '12px',
              padding: '1rem',
              textAlign: 'center',
              cursor: 'pointer',
              transition: 'all 0.3s ease',
              backgroundColor: isDragActive ? 'rgba(108, 99, 255, 0.1)' : 'transparent',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '8px',
              color: 'var(--text-secondary)'
            }}
          >
            <input {...getInputProps()} />
            <Plus size={20} color="var(--accent-primary)" />
            <span>Add more images</span>
          </div>
        </div>
      )}
      
      <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
        Supported formats: GeoTIFF, TIFF, PNG, JPEG
      </div>
    </div>
  );
};

export default UploadZone;
