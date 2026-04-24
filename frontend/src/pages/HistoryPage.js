import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { FRONTEND_CONFIG } from '../config';
import { getClasses } from '../services/api';

const BASE_URL = FRONTEND_CONFIG.apiUrl;
const COLORS   = { Normal: '#639922', Pneumonia: '#E24B4A', COVID19: '#BA7517' };
const RISK_BG  = { Normal: '#EAF3DE', Pneumonia: '#FCEBEB', COVID19: '#FAEEDA' };
const RISK_COL = { Normal: '#27500A', Pneumonia: '#791F1F', COVID19: '#633806' };

export default function HistoryPage() {
  const [history, setHistory]   = useState([]);
  const [selected, setSelected] = useState(null);
  const [loading, setLoading]   = useState(true);
  const [filter, setFilter]     = useState('All');
  const [classes, setClasses]   = useState([]);

  useEffect(() => {
    getClasses()
      .then(r => setClasses(r.classes || []))
      .catch(() => setClasses([]));

    axios.get(`${BASE_URL}${FRONTEND_CONFIG.apiPrefix}/feedback/history`)
      .then(r => { setHistory(r.data.history || []); setLoading(false); })
      .catch(() => { setHistory([]); setLoading(false); });
  }, []);

  const filtered = filter === 'All'
    ? history
    : history.filter(h => h.predicted_class === filter);

  const card  = { background:'#fff', borderRadius:16, padding:'1.5rem', boxShadow:'0 1px 3px rgba(0,0,0,0.08)', marginBottom:16 };
  const table = { width:'100%', borderCollapse:'collapse', fontSize:13 };
  const th    = { padding:'10px 12px', textAlign:'left', fontSize:11, fontWeight:500, textTransform:'uppercase', letterSpacing:'0.06em', color:'#5a7a94', borderBottom:'0.5px solid #e0e8f0' };
  const td    = { padding:'10px 12px', borderBottom:'0.5px solid #f0f4f8', color:'#1a1a2e', verticalAlign:'middle' };

  return (
    <div>
      <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center', marginBottom:'1.5rem' }}>
        <h2 style={{ fontFamily:'Syne,sans-serif', fontSize:24, color:'#0c2340' }}>Patient Scan History</h2>
        <div style={{ display:'flex', gap:8 }}>
          {['All', ...classes].map(f => (
            <button key={f} onClick={() => setFilter(f)} style={{
              padding:'6px 14px', borderRadius:999, fontSize:12, cursor:'pointer',
              background: filter===f ? '#0c2340' : '#fff',
              color: filter===f ? '#fff' : '#5a7a94',
              border: filter===f ? 'none' : '0.5px solid #ddd'
            }}>{f}</button>
          ))}
        </div>
      </div>

      <div style={{ display:'grid', gridTemplateColumns: selected ? '1fr 380px' : '1fr', gap:16 }}>

        <div style={card}>
          {loading ? (
            <p style={{ color:'#5a7a94', textAlign:'center', padding:'2rem' }}>Loading history...</p>
          ) : filtered.length === 0 ? (
            <p style={{ color:'#5a7a94', textAlign:'center', padding:'2rem' }}>
              No scan history yet. Upload an X-ray to get started.
            </p>
          ) : (
            <table style={table}>
              <thead>
                <tr>
                  <th style={th}>#</th>
                  <th style={th}>Timestamp</th>
                  <th style={th}>Prediction ID</th>
                  <th style={th}>Diagnosis</th>
                  <th style={th}>Confidence</th>
                  <th style={th}>Radiologist</th>
                  <th style={th}>Action</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((h, i) => (
                  <tr key={i} style={{ cursor:'pointer', background: selected?.prediction_id === h.prediction_id ? '#f0f7ff' : 'transparent' }}
                    onClick={() => setSelected(selected?.prediction_id === h.prediction_id ? null : h)}>
                    <td style={td}>{filtered.length - i}</td>
                    <td style={td}>{new Date(h.timestamp).toLocaleString()}</td>
                    <td style={{ ...td, fontFamily:'monospace', fontSize:11, color:'#5a7a94' }}>{h.prediction_id?.slice(0,12)}...</td>
                    <td style={td}>
                      <span style={{ background: RISK_BG[h.predicted_class], color: RISK_COL[h.predicted_class], padding:'3px 10px', borderRadius:999, fontSize:11, fontWeight:500 }}>
                        {h.predicted_class}
                      </span>
                    </td>
                    <td style={td}>
                      {h.confidence ? (
                        <div>
                          <div style={{ fontSize:12, fontWeight:500 }}>{(h.confidence*100).toFixed(1)}%</div>
                          <div style={{ height:4, width:80, background:'#f0f4f8', borderRadius:2, marginTop:3 }}>
                            <div style={{ height:4, width:`${h.confidence*80}px`, background: COLORS[h.predicted_class], borderRadius:2 }} />
                          </div>
                        </div>
                      ) : '—'}
                    </td>
                    <td style={td}>
                      {h.radiologist_confirmed === true  && <span style={{ color:'#27500A', fontSize:12 }}>✓ Confirmed</span>}
                      {h.radiologist_confirmed === false && <span style={{ color:'#791F1F', fontSize:12 }}>✗ Incorrect</span>}
                      {h.radiologist_confirmed === undefined && <span style={{ color:'#94b4cc', fontSize:12 }}>Pending</span>}
                    </td>
                    <td style={td}>
                      <button style={{ fontSize:11, padding:'4px 10px', borderRadius:6, border:'0.5px solid #ddd', background:'#fff', cursor:'pointer' }}>
                        {selected?.prediction_id === h.prediction_id ? 'Close' : 'View'}
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

        {selected && (
          <div style={{ ...card, alignSelf:'start' }}>
            <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center', marginBottom:16 }}>
              <span style={{ fontFamily:'Syne,sans-serif', fontSize:16, fontWeight:500, color:'#0c2340' }}>Scan Detail</span>
              <button onClick={() => setSelected(null)} style={{ border:'none', background:'none', cursor:'pointer', fontSize:18, color:'#94b4cc' }}>✕</button>
            </div>

            <div style={{ background: RISK_BG[selected.predicted_class], borderRadius:12, padding:'1rem', marginBottom:12 }}>
              <p style={{ fontSize:11, color: RISK_COL[selected.predicted_class], marginBottom:4 }}>Diagnosis</p>
              <p style={{ fontSize:24, fontWeight:600, fontFamily:'Syne,sans-serif', color:'#1a1a2e' }}>{selected.predicted_class}</p>
              {selected.confidence && <p style={{ fontSize:12, color:'#5a7a94', marginTop:4 }}>Confidence: {(selected.confidence*100).toFixed(1)}%</p>}
            </div>

            <div style={{ fontSize:12, color:'#5a7a94' }}>
              {[
                ['Time',        new Date(selected.timestamp).toLocaleString()],
                ['ID',          selected.prediction_id],
                ['Confirmed',   selected.radiologist_confirmed === true ? '✓ Yes' : selected.radiologist_confirmed === false ? '✗ No' : 'Pending'],
                ['Correct class', selected.correct_class || '—'],
                ['Comments',    selected.comments || '—'],
              ].map(([k,v]) => (
                <div key={k} style={{ display:'flex', justifyContent:'space-between', padding:'7px 0', borderBottom:'0.5px solid #f0f4f8' }}>
                  <span>{k}</span>
                  <span style={{ color:'#1a1a2e', fontWeight:500, textAlign:'right', maxWidth:180, wordBreak:'break-all' }}>{v}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      <div style={{ ...card, display:'grid', gridTemplateColumns:'repeat(4,1fr)', gap:12 }}>
        {[
          { label:'Total Scans',  value: history.length },
          ...classes.map((cls) => ({
            label: cls === 'COVID19' ? 'COVID-19' : cls,
            value: history.filter((h) => h.predicted_class === cls).length,
          })),
        ].map(({ label, value }) => (
          <div key={label} style={{ background:'#f0f4f8', borderRadius:12, padding:'1rem', textAlign:'center' }}>
            <p style={{ fontSize:28, fontWeight:500, color:'#0c2340' }}>{value}</p>
            <p style={{ fontSize:12, color:'#5a7a94', marginTop:4 }}>{label}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
