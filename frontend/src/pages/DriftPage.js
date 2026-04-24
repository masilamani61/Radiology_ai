import React, { useState, useEffect } from 'react';
import axios from 'axios';

const BASE = process.env.REACT_APP_API_URL || 'http://localhost:8005';

export default function DriftPage() {
  const [summary, setSummary]       = useState(null);
  const [selected, setSelected]     = useState(null);
  const [reportHtml, setReportHtml] = useState('');
  const [loading, setLoading]       = useState(true);

  useEffect(() => {
    axios.get(`${BASE}/api/v1/drift/summary`)
      .then(r => { setSummary(r.data); setLoading(false); })
      .catch(() => setLoading(false));
  }, []);

  const loadReport = (cls) => {
    setSelected(cls);
    axios.get(`${BASE}/api/v1/drift/report/${cls}`)
      .then(r => setReportHtml(r.data))
      .catch(() => setReportHtml('<p>Report not found. Run drift detection first.</p>'));
  };

  const card = { background:'#fff', borderRadius:16, padding:'1.5rem', boxShadow:'0 1px 3px rgba(0,0,0,0.08)', marginBottom:16 };

  return (
    <div>
      <h2 style={{ fontFamily:'Syne,sans-serif', fontSize:24, color:'#0c2340', marginBottom:'1.5rem' }}>
        Data Drift Detection
      </h2>

      {loading ? <p style={{ color:'#5a7a94' }}>Loading drift report...</p> : (
        <>
          <div style={{ ...card, display:'grid', gridTemplateColumns:'repeat(4,1fr)', gap:12, marginBottom:16 }}>
            <div style={{ background: summary?.any_drift ? '#FCEBEB' : '#EAF3DE', borderRadius:12, padding:'1rem', textAlign:'center' }}>
              <p style={{ fontSize:24, fontWeight:600, color: summary?.any_drift ? '#791F1F' : '#27500A' }}>
                {summary?.any_drift ? 'DRIFT' : 'STABLE'}
              </p>
              <p style={{ fontSize:11, color:'#5a7a94', marginTop:4 }}>Overall Status</p>
            </div>
            {['Normal','Pneumonia','COVID19'].map(cls => {
              const r = summary?.results?.[cls];
              return (
                <div key={cls} style={{ background: r?.drift_detected ? '#FCEBEB' : '#EAF3DE', borderRadius:12, padding:'1rem', textAlign:'center', cursor:'pointer' }}
                  onClick={() => loadReport(cls)}>
                  <p style={{ fontSize:20, fontWeight:600, color: r?.drift_detected ? '#791F1F' : '#27500A' }}>
                    {r?.drift_score?.toFixed(3) ?? '—'}
                  </p>
                  <p style={{ fontSize:11, color:'#5a7a94', marginTop:4 }}>{cls}</p>
                  <p style={{ fontSize:10, color: r?.drift_detected ? '#791F1F' : '#27500A' }}>
                    {r?.drift_detected ? '⚠ Drift' : '✓ Stable'}
                  </p>
                </div>
              );
            })}
          </div>

          <div style={card}>
            <p style={{ fontSize:11, color:'#5a7a94', textTransform:'uppercase', letterSpacing:'0.06em', marginBottom:12 }}>
              Evidently AI Report {selected ? `— ${selected}` : '(click a class above)'}
            </p>
            {selected ? (
              <iframe
                srcDoc={reportHtml}
                style={{ width:'100%', height:600, border:'none', borderRadius:8 }}
                title="Drift Report"
              />
            ) : (
              <div style={{ background:'#f0f4f8', borderRadius:8, padding:'3rem', textAlign:'center', color:'#5a7a94' }}>
                Click a class card above to view the Evidently AI drift report
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}
