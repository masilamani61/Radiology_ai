import React from 'react';

export default function Navbar({ page, setPage }) {
  const links = [
    { id:'upload',  label:'Analysis' },
    { id:'history', label:'History' },
    { id:'drift',   label:'Drift' },
    { id:'compare', label:'Models' },
    { id:'monitor', label:'Monitor' },
  ];
  return (
    <nav style={{ background:'#0c2340', padding:'0 2rem', display:'flex', alignItems:'center', justifyContent:'space-between', height:56 }}>
      <span style={{ color:'#fff', fontFamily:'Syne,sans-serif', fontSize:20, fontWeight:600, cursor:'pointer' }} onClick={() => setPage('upload')}>
        🫁 RadiologyAI
      </span>
      <div style={{ display:'flex', gap:4 }}>
        {links.map(({ id, label }) => (
          <button key={id} onClick={() => setPage(id)} style={{
            padding:'6px 14px', borderRadius:6, fontSize:13, cursor:'pointer', border:'none',
            background: page===id ? '#378ADD' : 'transparent',
            color: page===id ? '#fff' : '#94b4cc',
            fontFamily:'DM Sans,sans-serif'
          }}>{label}</button>
        ))}
      </div>
    </nav>
  );
}
