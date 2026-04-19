import React from 'react';

export default function Navbar({ page, setPage }) {
  const nav = { background: '#0c2340', padding: '0 2rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between', height: 56 };
  const logo = { color: '#fff', fontFamily: 'Syne, sans-serif', fontSize: 20, fontWeight: 600, cursor: 'pointer' };
  const links = { display: 'flex', gap: 8 };
  const link = (active) => ({
    padding: '6px 14px', borderRadius: 6, fontSize: 13, cursor: 'pointer', border: 'none',
    background: active ? '#378ADD' : 'transparent', color: active ? '#fff' : '#94b4cc', fontFamily: 'DM Sans, sans-serif'
  });
  return (
    <nav style={nav}>
      <span style={logo} onClick={() => setPage('upload')}>RadiologyAI</span>
      <div style={links}>
        <button style={link(page === 'upload')}     onClick={() => setPage('upload')}>Analysis</button>
        <button style={link(page === 'monitoring')} onClick={() => setPage('monitoring')}>ML Monitor</button>
      </div>
    </nav>
  );
}
