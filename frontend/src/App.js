import React, { useState } from 'react';
import { Toaster } from 'react-hot-toast';
import Navbar from './components/Navbar';
import UploadPage      from './pages/UploadPage';
import ResultPage      from './pages/ResultPage';
import MonitoringPage  from './pages/MonitoringPage';
import HistoryPage     from './pages/HistoryPage';
import DriftPage       from './pages/DriftPage';
import ModelComparePage from './pages/ModelComparePage';

export default function App() {
  const [page, setPage]     = useState('upload');
  const [result, setResult] = useState(null);
  const [image, setImage]   = useState(null);

  const handleResult = (res, img) => { setResult(res); setImage(img); setPage('result'); };
  const handleReset  = () => { setResult(null); setImage(null); setPage('upload'); };

  return (
    <div style={{ minHeight:'100vh', background:'#f0f4f8' }}>
      <Navbar page={page} setPage={setPage} />
      <main style={{ maxWidth:960, margin:'0 auto', padding:'2rem 1rem' }}>
        {page === 'upload'   && <UploadPage onResult={handleResult} />}
        {page === 'result'   && <ResultPage result={result} image={image} onReset={handleReset} />}
        {page === 'monitor'  && <MonitoringPage />}
        {page === 'history'  && <HistoryPage />}
        {page === 'drift'    && <DriftPage />}
        {page === 'compare'  && <ModelComparePage />}
      </main>
      <Toaster position="top-right" />
    </div>
  );
}
