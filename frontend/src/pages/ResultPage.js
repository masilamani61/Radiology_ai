import React, { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import toast from 'react-hot-toast';
import axios from "axios";
import { submitFeedback } from '../services/api';

const COLORS = { Normal: '#639922', Pneumonia: '#E24B4A', COVID19: '#BA7517' };
const RISK_COLORS = { Low: '#EAF3DE', High: '#FCEBEB' };
const RISK_TEXT   = { Low: '#27500A', High: '#791F1F' };

export default function ResultPage({ result, image, onReset }) {
  const [feedback, setFeedback] = useState('');
  const [submitted, setSubmitted] = useState(false);

  if (!result) return null;

  const chartData = result.all_probabilities.map(p => ({
    name: p.class_name, value: Math.round(p.confidence * 100)
  }));
  const handleDownloadPDF = async () => {
  try {
    const res = await axios.post(
      `${process.env.REACT_APP_API_URL || 'http://localhost:8005'}/api/v1/report/pdf`,
      { ...result, patient_id: 'PT-001', age: '' },
      { responseType: 'blob' }
    );
    const url  = window.URL.createObjectURL(new Blob([res.data]));
    const link = document.createElement('a');
    link.href  = url;
    link.setAttribute('download', 'RadiologyAI_Report.pdf');
    document.body.appendChild(link);
    link.click();
    link.remove();
    toast.success('PDF downloaded!');
  } catch(e) {
    toast.error('PDF generation failed');
  }
};

  const handleFeedback = async (confirmed) => {
    try {
      await submitFeedback({
        prediction_id        : `pred_${Date.now()}`,
        predicted_class      : result.predicted_class,
        correct_class        : confirmed ? result.predicted_class : feedback || null,
        radiologist_confirmed: confirmed,
        comments             : feedback,
      });
      setSubmitted(true);
      toast.success('Feedback submitted — thank you!');
    } catch {
      toast.error('Feedback failed');
    }
  };

  const card   = { background: '#fff', borderRadius: 16, padding: '1.5rem', boxShadow: '0 1px 3px rgba(0,0,0,0.08)', marginBottom: 16 };
  const badge  = (cls) => ({ display: 'inline-block', padding: '3px 10px', borderRadius: 999, fontSize: 11, fontWeight: 500, background: cls === 'Normal' ? '#EAF3DE' : '#FCEBEB', color: cls === 'Normal' ? '#27500A' : '#791F1F' });

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
        <h2 style={{ fontFamily: 'Syne, sans-serif', fontSize: 24, color: '#0c2340' }}>Analysis Result</h2>
        <button onClick={onReset} style={{ padding: '8px 16px', borderRadius: 8, border: '0.5px solid #ddd', background: '#fff', cursor: 'pointer', fontSize: 13 }}>New Scan</button>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>
        <div style={card}>
          <p style={{ fontSize: 11, color: '#5a7a94', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 8 }}>X-Ray Preview</p>
          <img src={image} alt="xray" style={{ width: '100%', borderRadius: 8, background: '#000', maxHeight: 220, objectFit: 'contain', display: 'block' }} />
        </div>

        <div style={card}>
          <p style={{ fontSize: 11, color: '#5a7a94', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 12 }}>Diagnosis</p>
          <div style={{ background: RISK_COLORS[result.risk_level], borderRadius: 12, padding: '1.25rem', marginBottom: 12 }}>
            <p style={{ fontSize: 11, color: RISK_TEXT[result.risk_level], marginBottom: 4 }}>Detected condition</p>
            <p style={{ fontSize: 28, fontWeight: 600, fontFamily: 'Syne, sans-serif', color: '#1a1a2e', marginBottom: 4 }}>{result.predicted_class}</p>
            <p style={{ fontSize: 13, color: '#5a7a94' }}>Confidence: {(result.confidence * 100).toFixed(1)}%</p>
            <div style={{ marginTop: 8, height: 6, background: 'rgba(0,0,0,0.08)', borderRadius: 3, overflow: 'hidden' }}>
              <div style={{ height: 6, width: `${result.confidence * 100}%`, background: COLORS[result.predicted_class], borderRadius: 3 }} />
            </div>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: '#5a7a94' }}>
            <span>Risk: <b style={{ color: RISK_TEXT[result.risk_level] }}>{result.risk_level}</b></span>
            <span>Inference: <b>{result.inference_time_ms.toFixed(0)}ms</b></span>
          </div>
        </div>
      </div>

      <div style={card}>
        <p style={{ fontSize: 11, color: '#5a7a94', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 12 }}>Class probabilities</p>
        <ResponsiveContainer width="100%" height={140}>
          <BarChart data={chartData} layout="vertical" margin={{ left: 20 }}>
            <XAxis type="number" domain={[0, 100]} tickFormatter={v => `${v}%`} fontSize={11} />
            <YAxis type="category" dataKey="name" fontSize={12} width={80} />
            <Tooltip formatter={v => `${v}%`} />
            <Bar dataKey="value" radius={[0, 4, 4, 0]}>
              {chartData.map((d, i) => <Cell key={i} fill={COLORS[d.name]} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {result.gradcam_base64 && (
        <div style={card}>
          <p style={{ fontSize: 11, color: '#5a7a94', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 12 }}>Grad-CAM Heatmap — model attention regions</p>
          <img src={`data:image/png;base64,${result.gradcam_base64}`} alt="gradcam" style={{ width: '100%', borderRadius: 8, display: 'block' }} />
          <p style={{ fontSize: 11, color: '#94b4cc', marginTop: 8 }}>Red regions indicate areas that most influenced the prediction</p>
        </div>
      )}

      <div style={card}>
        <p style={{ fontSize: 11, color: '#5a7a94', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 12 }}>Radiologist Feedback</p>
        {submitted ? (
          <p style={{ color: '#27500A', fontSize: 14 }}>✓ Feedback submitted successfully</p>
        ) : (
          <div>
            <input placeholder="Optional: correct diagnosis or comments" value={feedback} onChange={e => setFeedback(e.target.value)}
              style={{ width: '100%', padding: '8px 12px', borderRadius: 8, border: '0.5px solid #ddd', fontSize: 13, marginBottom: 12 }} />
            <div style={{ display: 'flex', gap: 8 }}>
              <button onClick={() => handleFeedback(true)}  style={{ flex: 1, padding: 10, background: '#EAF3DE', color: '#27500A', border: 'none', borderRadius: 8, cursor: 'pointer', fontSize: 13 }}>✓ Confirm Prediction</button>
              <button onClick={() => handleFeedback(false)} style={{ flex: 1, padding: 10, background: '#FCEBEB', color: '#791F1F', border: 'none', borderRadius: 8, cursor: 'pointer', fontSize: 13 }}>✗ Incorrect</button>
            </div>
          </div>
        )}
      </div>
      <div style={{ display:'flex', gap:8 }}>
  <button onClick={handleDownloadPDF} style={{ flex:1, padding:10, background:'#0c2340', color:'#fff', border:'none', borderRadius:8, cursor:'pointer', fontSize:13 }}>
    📄 Download PDF Report
  </button>
  <button onClick={onReset} style={{ flex:1, padding:10, background:'#f0f4f8', color:'#5a7a94', border:'none', borderRadius:8, cursor:'pointer', fontSize:13 }}>
    New Scan
  </button>
</div>

      <p style={{ fontSize: 11, color: '#94b4cc', textAlign: 'center', marginTop: 12 }}>
        {result.disclaimer}
      </p>
    </div>
  );
}
const handleDownloadPDF = async () => {
  try {
    const res = await axios.post('http://localhost:8005/api/v1/report/pdf',
      { ...result, patient_id: 'PT-001', age: '' },
      { responseType: 'blob' }
    );
    const url  = window.URL.createObjectURL(new Blob([res.data]));
    const link = document.createElement('a');
    link.href  = url;
    link.setAttribute('download', 'RadiologyAI_Report.pdf');
    document.body.appendChild(link);
    link.click();
  } catch(e) { toast.error('PDF generation failed'); }
};
