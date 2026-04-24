import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, Legend } from 'recharts';

const BASE = process.env.REACT_APP_API_URL || 'http://localhost:8005';

export default function ModelComparePage() {
  const [runs, setRuns]       = useState([]);
  const [loading, setLoading] = useState(true);
  const [metric, setMetric]   = useState('val_f1');

  useEffect(() => {
    axios.get(`${BASE}/api/v1/mlflow/runs`)
      .then(r => { setRuns(r.data.runs || []); setLoading(false); })
      .catch(() => setLoading(false));
  }, []);

  const card  = { background:'#fff', borderRadius:16, padding:'1.5rem', boxShadow:'0 1px 3px rgba(0,0,0,0.08)', marginBottom:16 };
  const th    = { padding:'10px 12px', textAlign:'left', fontSize:11, fontWeight:500, textTransform:'uppercase', letterSpacing:'0.06em', color:'#5a7a94', borderBottom:'0.5px solid #e0e8f0', background:'#f8fafc' };
  const td    = { padding:'10px 12px', borderBottom:'0.5px solid #f0f4f8', fontSize:12, color:'#1a1a2e' };
  const COLORS = ['#0C447C','#378ADD','#639922','#BA7517','#E24B4A'];

  const chartData = runs.map((r, i) => ({
    name    : `Run ${i+1}`,
    run_id  : r.run_id,
    val_f1  : r.metrics.val_f1,
    val_acc : r.metrics.val_acc,
    test_f1 : r.metrics.test_f1,
    test_acc: r.metrics.test_acc,
    macro_auc: r.metrics.macro_auc,
  }));

  const bestRun = runs.reduce((best, r) =>
    (r.metrics.val_f1 > (best?.metrics?.val_f1 || 0)) ? r : best, null);

  return (
    <div>
      <h2 style={{ fontFamily:'Syne,sans-serif', fontSize:24, color:'#0c2340', marginBottom:'1.5rem' }}>
        MLflow Model Comparison
      </h2>

      {loading ? <p style={{ color:'#5a7a94' }}>Loading runs...</p> : runs.length === 0 ? (
        <div style={card}>
          <p style={{ color:'#5a7a94', textAlign:'center', padding:'2rem' }}>
            No MLflow runs found. Start training to see experiments here.
          </p>
        </div>
      ) : (
        <>
          {bestRun && (
            <div style={{ ...card, background:'#EAF3DE', border:'0.5px solid #97C459' }}>
              <p style={{ fontSize:11, color:'#27500A', textTransform:'uppercase', letterSpacing:'0.06em', marginBottom:4 }}>Best Model</p>
              <div style={{ display:'flex', gap:24, alignItems:'center' }}>
                <div>
                  <p style={{ fontSize:20, fontWeight:600, color:'#27500A' }}>Run {bestRun.run_id}</p>
                  <p style={{ fontSize:12, color:'#3B6D11' }}>val_f1={bestRun.metrics.val_f1} · val_acc={bestRun.metrics.val_acc}</p>
                </div>
                <div style={{ display:'flex', gap:16 }}>
                  {Object.entries(bestRun.metrics).map(([k,v]) => v > 0 && (
                    <div key={k} style={{ textAlign:'center' }}>
                      <p style={{ fontSize:16, fontWeight:500, color:'#27500A' }}>{v}</p>
                      <p style={{ fontSize:10, color:'#5a7a94' }}>{k}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          <div style={card}>
            <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center', marginBottom:16 }}>
              <p style={{ fontSize:13, fontWeight:500, color:'#0c2340' }}>Metric Comparison</p>
              <select value={metric} onChange={e => setMetric(e.target.value)}
                style={{ padding:'6px 12px', borderRadius:8, border:'0.5px solid #ddd', fontSize:12 }}>
                {['val_f1','val_acc','test_f1','test_acc','macro_auc'].map(m => (
                  <option key={m} value={m}>{m}</option>
                ))}
              </select>
            </div>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={chartData}>
                <XAxis dataKey="name" fontSize={12} />
                <YAxis domain={[0,1]} fontSize={12} />
                <Tooltip formatter={v => v.toFixed(4)} />
                <Bar dataKey={metric} radius={[4,4,0,0]}>
                  {chartData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div style={card}>
            <p style={{ fontSize:11, color:'#5a7a94', textTransform:'uppercase', letterSpacing:'0.06em', marginBottom:12 }}>All Runs</p>
            <table style={{ width:'100%', borderCollapse:'collapse', fontSize:12 }}>
              <thead>
                <tr>
                  {['Run ID','Time','Batch','Epochs P1','Epochs P2','LR P1','val_f1','val_acc','test_f1','macro_auc','Git'].map(h => (
                    <th key={h} style={th}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {runs.map((r, i) => (
                  <tr key={i} style={{ background: r.run_id === bestRun?.run_id ? '#f0f9f0' : 'transparent' }}>
                    <td style={{ ...td, fontFamily:'monospace', fontSize:11 }}>{r.run_id}</td>
                    <td style={td}>{r.start_time}</td>
                    <td style={td}>{r.params.batch_size}</td>
                    <td style={td}>{r.params.epochs_phase1}</td>
                    <td style={td}>{r.params.epochs_phase2}</td>
                    <td style={td}>{r.params.lr_phase1}</td>
                    <td style={{ ...td, fontWeight:500, color:'#0c2340' }}>{r.metrics.val_f1}</td>
                    <td style={td}>{r.metrics.val_acc}</td>
                    <td style={td}>{r.metrics.test_f1}</td>
                    <td style={td}>{r.metrics.macro_auc}</td>
                    <td style={{ ...td, fontFamily:'monospace', fontSize:10 }}>{r.params.git_commit}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
