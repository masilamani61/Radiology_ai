import React, { useEffect, useState } from 'react';
import { FRONTEND_CONFIG } from '../config';
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
    { label: 'Model',         value: health?.model_name || 'Unknown', color: '#0C447C', bg: '#E6F1FB' },
    { label: 'Framework',     value: health?.framework || 'Unknown',  color: '#0C447C', bg: '#E6F1FB' },
    { label: 'Classes',       value: health?.class_count ?? '-',      color: '#633806', bg: '#FAEEDA' },
    { label: 'Image Size',    value: health?.image_size ? `${health.image_size}x${health.image_size}` : '-', color: '#633806', bg: '#FAEEDA' },
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
         View Grafana Dashboard at <a href={FRONTEND_CONFIG.grafanaUrl || '#'} target="_blank" rel="noreferrer" style={{ color: '#378ADD' }}>{FRONTEND_CONFIG.grafanaUrl || 'Not configured'}</a>

        </div>
      </div>

      <div style={card}>
        <p style={label}>MLflow Experiments</p>
        <div style={{ background: '#f0f4f8', borderRadius: 8, padding: '2rem', textAlign: 'center', color: '#5a7a94', fontSize: 13 }}>
          View training runs at <a href={FRONTEND_CONFIG.mlflowUrl || '#'} target="_blank" rel="noreferrer" style={{ color: '#378ADD' }}>{FRONTEND_CONFIG.mlflowUrl || 'Not configured'}</a>
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
