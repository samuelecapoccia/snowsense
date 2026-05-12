import { useState, useRef } from 'react';
import SearchBar from './components/SearchBar';
import ResortCard from './components/ResortCard';
import SkeletonCard from './components/SkeletonCard';
import ClarificationBubble from './components/ClarificationBubble';
import { useRecommend } from './hooks/useRecommend';
import './App.css';

const NAV_LINKS = ['Resort'];

function Snowflake() {
  return (
    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <line x1="12" y1="2" x2="12" y2="22" />
      <line x1="2" y1="12" x2="22" y2="12" />
      <line x1="5" y1="5" x2="19" y2="19" />
      <line x1="19" y1="5" x2="5" y2="19" />
      <polyline points="12 6 10 4 12 2 14 4 12 6" />
      <polyline points="12 18 10 20 12 22 14 20 12 18" />
      <polyline points="6 12 4 10 2 12 4 14 6 12" />
      <polyline points="18 12 20 10 22 12 20 14 18 12" />
    </svg>
  );
}

const FLAKES = Array.from({ length: 60 }, (_, i) => {
  const dur = 8 + (i % 7) * 1.4;
  const startPct = (i * 17.3 + 5) % 100;
  return {
    left:  (i * 7.3 + 2.1) % 100,
    size:  7 + (i % 5) * 2,
    dur,
    delay: -(startPct / 100 * dur),
    drift: ((i % 9) - 4) * 5,
    op:    0.14 + (i % 5) * 0.055,
  };
});

const DECO_RESORTS = [
  { name: 'Zermatt', country: '🇨🇭 Switzerland', match: 95, price: 65, slope: 360, alt: 3883 },
  { name: 'Livigno', country: '🇮🇹 Italy',       match: 87, price: 42, slope: 115, alt: 3002 },
  { name: 'Les Arcs', country: '🇫🇷 France',     match: 79, price: 55, slope: 200, alt: 3226 },
];

function DecoCard({ resort }: { resort: typeof DECO_RESORTS[0] }) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const rectRef = useRef<DOMRect | null>(null);
  const circumference = 100;
  const filled = resort.match;

  const onMouseEnter = () => {
    const el = wrapRef.current;
    if (!el) return;
    rectRef.current = el.getBoundingClientRect(); // capture untransformed bounds
    el.style.transition = 'none';
  };

  const onMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    const el = wrapRef.current;
    const rect = rectRef.current;
    if (!el || !rect) return;
    const x = (e.clientX - rect.left) / rect.width - 0.5;
    const y = (e.clientY - rect.top)  / rect.height - 0.5;
    el.style.transform = `translateY(${y * -14}px) translateX(${x * 10}px) rotate(${x * 7}deg) scale(1.03)`;
  };

  const onMouseLeave = () => {
    const el = wrapRef.current;
    if (!el) return;
    rectRef.current = null;
    el.style.transition = 'transform 0.5s cubic-bezier(0.34, 1.56, 0.64, 1)';
    el.style.transform = '';
    setTimeout(() => { if (el) el.style.transition = ''; }, 500);
  };

  return (
    <div ref={wrapRef} className="deco-card-wrapper"
      onMouseEnter={onMouseEnter} onMouseMove={onMouseMove} onMouseLeave={onMouseLeave}>
    <div className="deco-card">
      <div className="deco-card-header">
        <div className="deco-card-name">
          <span className="deco-resort-name">{resort.name}</span>
          <span className="deco-country">{resort.country}</span>
        </div>
        <div className="match-circle">
          <svg viewBox="0 0 36 36" className="donut">
            <circle className="donut-bg" cx="18" cy="18" r="15.9" />
            <circle
              className="donut-fill" cx="18" cy="18" r="15.9"
              strokeDasharray={`${filled} ${circumference - filled}`}
              strokeDashoffset="25"
            />
          </svg>
          <span className="match-pct">{resort.match}%</span>
        </div>
      </div>
      <div className="deco-stats">
        <span>€{resort.price}/giorno</span>
        <span>{resort.slope} km</span>
        <span>{resort.alt} m</span>
      </div>
    </div>
    </div>
  );
}

export default function App() {
  const { state, search, reset } = useRecommend();
  const [activeNav, setActiveNav] = useState('');
  const [lastQuery, setLastQuery] = useState('');

  const isIdle = state.phase === 'idle';

  const handleSearch = (q: string) => {
    setActiveNav('');
    setLastQuery(q);
    search(q);
  };

  const scrollToSearch = () => {
    document.querySelector('.search-section')?.scrollIntoView({ behavior: 'smooth' });
  };

  return (
    <div className="app">
      {/* Snowflakes */}
      <div className="snow-layer" aria-hidden="true">
        {FLAKES.map((f, i) => (
          <span key={i} className="flake" style={{
            left: `${f.left}%`,
            fontSize: `${f.size}px`,
            '--dur': `${f.dur}s`,
            '--delay': `${f.delay}s`,
            '--drift': `${f.drift}px`,
            '--flake-op': f.op,
          } as React.CSSProperties}>❄</span>
        ))}
      </div>

      {/* Mountain silhouette */}
      <div className="mountain-bg" aria-hidden="true" />

      {/* Navbar */}
      <nav className="navbar">
        <div className="nav-brand" onClick={reset} style={{ cursor: 'pointer' }}>
          <Snowflake />
          <span>SnowSense</span>
        </div>
        <div className="nav-links">
          {NAV_LINKS.map(l => (
            <button
              key={l}
              className={`nav-link ${activeNav === l ? 'active' : ''}`}
              onClick={() => { setActiveNav(l); scrollToSearch(); }}
            >{l}</button>
          ))}
        </div>
      </nav>

      {/* Hero — idle: two-column asymmetric */}
      {isIdle && (
        <main className="hero hero--idle">
          <div className="hero-left">
            <h1 className="hero-title">
              Trova la tua<br />
              <em>pista perfetta</em>
            </h1>

            <p className="hero-subtitle">
              Descrivi cosa cerchi: la nostra AI abbina la tua richiesta ai migliori resort sciistici europei e spiega perché.
            </p>

            <SearchBar onSearch={handleSearch} loading={false} />

            <div className="stats-row stats-row--left">
              <div className="stat-item">
                <span className="stat-big">372</span>
                <span className="stat-desc">RESORT INDICIZZATI</span>
              </div>
              <div className="stat-item">
                <span className="stat-big">18</span>
                <span className="stat-desc">PAESI</span>
              </div>
              <div className="stat-item">
                <span className="stat-big">3</span>
                <span className="stat-desc">CONSIGLI PERSONALIZZATI</span>
              </div>
            </div>
          </div>

          <div className="hero-right" aria-hidden="true">
            {DECO_RESORTS.map((r) => (
              <DecoCard key={r.name} resort={r} />
            ))}
          </div>
        </main>
      )}

      {/* Hero — active: centered column */}
      {!isIdle && (
        <main className="hero hero--active">
          <SearchBar onSearch={handleSearch} loading={state.phase === 'loading'} initialValue={lastQuery} />

          {state.phase === 'clarification' && (
            <ClarificationBubble
              message={state.message}
              onReply={handleSearch}
              loading={false}
            />
          )}

          {state.phase === 'loading' && (
            <section className="results-section">
              <div className="results-grid">
                <SkeletonCard featured />
                <SkeletonCard />
                <SkeletonCard />
              </div>
            </section>
          )}

          {state.phase === 'results' && (
            <section className="results-section">
              <div className="results-header">
                <h2>Resort consigliati</h2>
                <p className="results-meta">
                  {state.data.resorts.length} risultati per "{state.query}"
                </p>
                <button className="new-search-btn" onClick={reset}>← Ricerca</button>
              </div>
              {state.data.resorts.length === 0 ? (
                <div className="empty-state">
                  <span className="empty-icon">🏔</span>
                  <h3>Nessun resort trovato</h3>
                  <p>Prova a modificare la ricerca — meno filtri o una zona diversa.</p>
                  <button className="search-btn" onClick={reset}>Nuova ricerca</button>
                </div>
              ) : (
                <div className="results-grid">
                  {state.data.resorts.map((r, i) => (
                    <ResortCard key={r.resort_id} resort={r} rank={i + 1} />
                  ))}
                </div>
              )}
            </section>
          )}

          {state.phase === 'error' && (
            <div className="error-bubble">
              <p>⚠ {state.message}</p>
              <button onClick={reset}>Riprova</button>
            </div>
          )}
        </main>
      )}
    </div>
  );
}
