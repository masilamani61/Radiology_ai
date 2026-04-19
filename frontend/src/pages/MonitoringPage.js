import React, { useEffect, useState } from 'react';
import { checkHealth } from '../services/api';

export default function MonitoringPage() {
  const [health, setHealth] = useState(null);

  useEffect(() => {
    checkHealth().then(setHealth).catch(() => setHealth({ status: 'error', model_loaded: false }));
  }, []);

  const card  = { background: '#fff', borderRadius: 16, padding: '1.5rem', boxShadow: '0 1px 3px rgba(0,0,0,0.08)', marginBottom: 16 };
  const label = { fontSize: 11, color: '#5a7a94', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 8 };

  const metrics = [
    { label: 'API Status',    value: health?.status === 'ok' ? 'Online' : 'Error',   color: health?.status === 'ok' ? '#27500A' : '#791F1F', bg: health?.status === 'ok' ? '#EAF3DE' : '#FCEBEB' },
    { label: 'Model Loaded',  value: health?.model_loaded ? 'Yes' : 'No',            color: health?.model_loaded ? '#27500A' : '#791F1F',    bg: health?.model_loaded ? '#EAF3DE' : '#FCEBEB'    },
    { label: 'Model',         value: 'EfficientNetB0', color: '#0C447C', bg: '#E6F1FB' },
    { label: 'Framework',     value: 'PyTorch 2.3',    color: '#0C447C', bg: '#E6F1FB' },
    { label: 'Classes',       value: '3',              color: '#633806', bg: '#FAEEDA' },
    { label: 'Image Size',    value: '224×224',        color: '#633806', bg: '#FAEEDA' },
  ];

  return (
    <div>
      <h2 style={{ fontFamily: 'Syne, sans-serif', fontSize: 24, color: '#0c2340', marginBottom: '1.5rem' }}>ML Pipeline Monitor</h2>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12, marginBottom: 16 }}>
        {metrics.map((m, i) => (
          <div key={i} style={{ background: m.bg, borderRadius: 12, padding: '1rem' }}>
            <p style={{ fontSize: 11, color: m.color, opacity: 0.7, marginBottom: 4 }}>{m.label}</p>
            <p style={{ fontSize: 18, fontWeight: 500, color: m.color }}>{m.value}</p>
          </div>
        ))}
      </div>

      <div style={card}>
        <p style={label}>Grafana Dashboard</p>
        <div style={{ background: '#f0f4f8', borderRadius: 8, padding: '2rem', textAlign: 'center', color: '#5a7a94', fontSize: 13 }}>
          Start Grafana on port 3001 to see live metrics here.<br/>
          <code style={{ fontSize: 12, background: '#e0e8f0', padding: '2px 6px', borderRadius: 4 }}>docker compose up grafana</code>
        </div>
      </div>

      <div style={card}>
        <p style={label}>MLflow Experiments</p>
        <div style={{ background: '#f0f4f8', borderRadius: 8, padding: '2rem', textAlign: 'center', color: '#5a7a94', fontSize: 13 }}>
          View training runs at <a href="http://localhost:5001" target="_blank" rel="noreferrer" style={{ color: '#378ADD' }}>http://localhost:5001</a>
        </div>
      </div>

      <div style={card}>
        <p style={label}>DVC Pipeline</p>
        <div style={{ fontFamily: 'monospace', fontSize: 12, background: '#0c2340', color: '#94b4cc', padding: '1rem', borderRadius: 8, lineHeight: 2 }}>
          <span style={{ color: '#639922' }}>download</span> → <span style={{ color: '#639922' }}>validate</span> → <span style={{ color: '#639922' }}>preprocess</span> → <span style={{ color: '#378ADD' }}>train</span> → <span style={{ color: '#378ADD' }}>evaluate</span> → <span style={{ color: '#EF9F27' }}>explain</span> → <span style={{ color: '#EF9F27' }}>export</span>
        </div>
      </div>
    </div>
  );
}
