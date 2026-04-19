import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import toast from 'react-hot-toast';
import { predictXray } from '../services/api';

export default function UploadPage({ onResult }) {
  const [file, setFile]       = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [patientId, setPatientId] = useState('');
  const [age, setAge]         = useState('');

  const onDrop = useCallback((accepted) => {
    if (!accepted.length) return;
    const f = accepted[0];
    setFile(f);
    setPreview(URL.createObjectURL(f));
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop, accept: { 'image/jpeg': [], 'image/png': [] }, maxFiles: 1
  });

  const handleAnalyse = async () => {
    if (!file) { toast.error('Please upload an X-ray image first'); return; }
    setLoading(true);
    try {
      const result = await predictXray(file);
      toast.success('Analysis complete!');
      onResult(result, preview);
    } catch (e) {
      toast.error(e.response?.data?.detail || 'Analysis failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const card = { background: '#fff', borderRadius: 16, padding: '2rem', boxShadow: '0 1px 3px rgba(0,0,0,0.08)', marginBottom: 16 };
  const input = { width: '100%', padding: '8px 12px', borderRadius: 8, border: '0.5px solid #ddd', fontSize: 13, fontFamily: 'DM Sans, sans-serif', marginTop: 4 };
  const btn = { width: '100%', padding: 12, background: loading ? '#94b4cc' : '#0c2340', color: '#fff', border: 'none', borderRadius: 10, fontSize: 14, fontFamily: 'Syne, sans-serif', fontWeight: 500, cursor: loading ? 'not-allowed' : 'pointer', marginTop: 8 };

  return (
    <div>
      <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
        <h1 style={{ fontFamily: 'Syne, sans-serif', fontSize: 32, color: '#0c2340', marginBottom: 8 }}>Chest X-Ray Analysis</h1>
        <p style={{ color: '#5a7a94', fontSize: 15 }}>AI-powered triage support for clinicians — Normal · Pneumonia · COVID-19</p>
      </div>

      <div style={card}>
        <div {...getRootProps()} style={{
          border: `2px dashed ${isDragActive ? '#378ADD' : preview ? '#378ADD' : '#ccd9e3'}`,
          borderRadius: 12, padding: preview ? 0 : '3rem 2rem', textAlign: 'center',
          cursor: 'pointer', background: isDragActive ? '#f0f7ff' : '#fafcff', overflow: 'hidden', transition: 'all 0.2s'
        }}>
          <input {...getInputProps()} />
          {preview ? (
            <div>
              <img src={preview} alt="xray" style={{ width: '100%', maxHeight: 280, objectFit: 'contain', background: '#000', display: 'block' }} />
              <div style={{ padding: '8px 12px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: 12, color: '#5a7a94', fontFamily: 'monospace' }}>{file?.name}</span>
                <span style={{ fontSize: 12, color: '#94b4cc' }}>{(file?.size / 1024 / 1024).toFixed(1)} MB</span>
              </div>
            </div>
          ) : (
            <div>
              <div style={{ fontSize: 48, marginBottom: 12 }}>🫁</div>
              <p style={{ color: '#0c2340', fontWeight: 500, marginBottom: 4 }}>Drop chest X-ray here</p>
              <p style={{ color: '#94b4cc', fontSize: 13 }}>JPEG or PNG · max 10MB</p>
            </div>
          )}
        </div>
      </div>

      <div style={{ ...card, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
        <div>
          <label style={{ fontSize: 11, color: '#5a7a94', textTransform: 'uppercase', letterSpacing: '0.06em' }}>Patient ID</label>
          <input style={input} value={patientId} onChange={e => setPatientId(e.target.value)} placeholder="PT-000001" />
        </div>
        <div>
          <label style={{ fontSize: 11, color: '#5a7a94', textTransform: 'uppercase', letterSpacing: '0.06em' }}>Age</label>
          <input style={input} value={age} onChange={e => setAge(e.target.value)} placeholder="42" type="number" />
        </div>
      </div>

      <button style={btn} onClick={handleAnalyse} disabled={loading}>
        {loading ? 'Analysing...' : 'Analyse X-Ray'}
      </button>

      <p style={{ fontSize: 11, color: '#94b4cc', textAlign: 'center', marginTop: 12, lineHeight: 1.6 }}>
        AI-assisted preliminary assessment only. Not a substitute for professional radiological diagnosis.
      </p>
    </div>
  );
}
