import type { Resort } from '../types/api';

const FLAG: Record<string, string> = {
  Italy: '🇮🇹', France: '🇫🇷', Austria: '🇦🇹', Switzerland: '🇨🇭',
  Germany: '🇩🇪', Spain: '🇪🇸', Andorra: '🇦🇩', Norway: '🇳🇴',
  Sweden: '🇸🇪', Finland: '🇫🇮', Poland: '🇵🇱', Bulgaria: '🇧🇬',
  Slovenia: '🇸🇮', Slovakia: '🇸🇰', 'Czech Republic': '🇨🇿',
  'United Kingdom': '🇬🇧', 'Southern Russia': '🇷🇺', Siberia: '🇷🇺',
};

interface Props {
  resort: Resort;
  rank: number;
}

export default function ResortCard({ resort, rank }: Props) {
  const flag = FLAG[resort.Country] ?? '🏔';
  const total = Math.max(resort.TotalSlope, 1);
  const begPct = Math.round((resort.BeginnerSlope / total) * 100);
  const intPct = Math.round((resort.IntermediateSlope / total) * 100);
  const difPct = Math.round((resort.DifficultSlope / total) * 100);
  const hasSnowpark = resort.Snowparks?.toLowerCase() === 'yes';
  const hasNightSki = resort.NightSki?.toLowerCase() === 'yes';
  const matchPct = Math.round(resort.scores.final * 100);

  const featured = rank === 1;

  const header = (
    <div className="card-header">
      <div className="rank-badge">#{rank}</div>
      <div className="card-title-block">
        <h3 className="resort-name">{resort.Resort}</h3>
        <p className="resort-country">{flag} {resort.Country}</p>
      </div>
      <div className="match-circle">
        <svg viewBox="0 0 36 36" className="donut">
          <circle className="donut-bg" cx="18" cy="18" r="15.9" />
          <circle
            className="donut-fill"
            cx="18" cy="18" r="15.9"
            strokeDasharray={`${matchPct} ${100 - matchPct}`}
            strokeDashoffset="25"
          />
        </svg>
        <span className="match-pct">{matchPct}%</span>
      </div>
    </div>
  );

  const badges = (
    <div className="badges">
      {hasSnowpark && <span className="badge">🏂 Snowpark</span>}
      {hasNightSki && <span className="badge">🌙 Sci notturno</span>}
      <span className="badge">🚡 {resort.TotalLifts} impianti</span>
    </div>
  );

  const stats = (
    <div className="card-stats">
      <div className="stat">
        <span className="stat-value">{resort.DayPassPriceAdult}€</span>
        <span className="stat-label">Ski pass</span>
      </div>
      <div className="stat">
        <span className="stat-value">{resort.TotalSlope}km</span>
        <span className="stat-label">Piste totali</span>
      </div>
      <div className="stat">
        <span className="stat-value">{resort.HighestPoint}m</span>
        <span className="stat-label">Quota max</span>
      </div>
      {resort.distance_km !== null && resort.distance_km !== undefined && (
        <div className="stat">
          <span className="stat-value">{resort.distance_km}km</span>
          <span className="stat-label">Distanza</span>
        </div>
      )}
    </div>
  );

  const slopeBar = (
    <div className="slope-bar-section">
      <div className="slope-bar">
        <div className="slope-seg beginner"     style={{ width: `${begPct}%` }} title={`Facili ${begPct}%`} />
        <div className="slope-seg intermediate" style={{ width: `${intPct}%` }} title={`Intermedie ${intPct}%`} />
        <div className="slope-seg difficult"    style={{ width: `${difPct}%` }} title={`Difficili ${difPct}%`} />
      </div>
      <div className="slope-legend">
        <span className="leg beginner">● Facili {begPct}%</span>
        <span className="leg intermediate">● Intermedie {intPct}%</span>
        <span className="leg difficult">● Difficili {difPct}%</span>
      </div>
    </div>
  );

  const explanation = resort.explanation && (
    <div className="explanation"><p>{resort.explanation}</p></div>
  );

  const explorationReason = resort.is_exploration && resort.exploration_reason && (
    <div className="exploration-reason">{resort.exploration_reason}</div>
  );

  return (
    <div className={`resort-card ${featured ? 'resort-card--featured' : ''} ${resort.is_exploration ? 'exploration' : ''}`}>
      {resort.is_exploration && (
        <div className="exploration-badge">Scoperta</div>
      )}

      {featured ? (
        <>
          <div className="card-left">
            {header}
            {badges}
          </div>
          <div className="card-right">
            {stats}
            {slopeBar}
            {explanation}
            {explorationReason}
          </div>
        </>
      ) : (
        <>
          {header}
          {stats}
          {slopeBar}
          {badges}
          {explanation}
          {explorationReason}
        </>
      )}
    </div>
  );
}
