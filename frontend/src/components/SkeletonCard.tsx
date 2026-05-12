interface Props { featured?: boolean; }

export default function SkeletonCard({ featured = false }: Props) {
  if (featured) {
    return (
      <div className="resort-card resort-card--featured skeleton-card">
        <div className="card-left">
          {/* header */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <div className="skeleton" style={{ width: 32, height: 32, borderRadius: '50%', flexShrink: 0 }} />
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 6 }}>
              <div className="skeleton" style={{ height: 22, width: '80%' }} />
              <div className="skeleton" style={{ height: 12, width: '50%' }} />
            </div>
          </div>
          {/* badges */}
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            <div className="skeleton" style={{ height: 28, width: 90, borderRadius: 99 }} />
            <div className="skeleton" style={{ height: 28, width: 110, borderRadius: 99 }} />
            <div className="skeleton" style={{ height: 28, width: 80, borderRadius: 99 }} />
          </div>
        </div>
        <div className="card-right">
          {/* stats */}
          <div style={{ display: 'flex', gap: 16 }}>
            {[1,2,3,4].map(i => (
              <div key={i} style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 4 }}>
                <div className="skeleton" style={{ height: 26 }} />
                <div className="skeleton" style={{ height: 10, width: '60%' }} />
              </div>
            ))}
          </div>
          {/* slope bar */}
          <div className="skeleton" style={{ height: 6, borderRadius: 99 }} />
          {/* explanation */}
          <div className="skeleton skeleton-text" />
          <div className="skeleton skeleton-text" />
          <div className="skeleton skeleton-text short" />
        </div>
      </div>
    );
  }

  return (
    <div className="resort-card skeleton-card">
      {/* header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        <div className="skeleton" style={{ width: 32, height: 32, borderRadius: '50%', flexShrink: 0 }} />
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 6 }}>
          <div className="skeleton skeleton-title" style={{ marginBottom: 0 }} />
          <div className="skeleton skeleton-subtitle" style={{ marginBottom: 0 }} />
        </div>
        <div className="skeleton" style={{ width: 48, height: 48, borderRadius: '50%', flexShrink: 0 }} />
      </div>
      {/* stats */}
      <div style={{ display: 'flex', gap: 12 }}>
        {[1,2,3].map(i => (
          <div key={i} style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 4 }}>
            <div className="skeleton" style={{ height: 22 }} />
            <div className="skeleton" style={{ height: 10, width: '60%' }} />
          </div>
        ))}
      </div>
      {/* slope bar */}
      <div className="skeleton skeleton-line" style={{ height: 6 }} />
      {/* explanation */}
      <div className="skeleton skeleton-text" />
      <div className="skeleton skeleton-text short" />
    </div>
  );
}
