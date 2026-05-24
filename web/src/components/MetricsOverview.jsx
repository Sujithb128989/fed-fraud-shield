import React from 'react'
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'

const MetricsOverview = ({ data }) => {
  if (!data || !data.averages) return <div className="glass-panel">Initializing global model...</div>

  const { roc_auc, f1_score, precision } = data.averages

  // Format data for Recharts
  const chartData = data.history.map(h => ({
    round: `Round ${h.round_number}`,
    roc_auc: (h.roc_auc * 100).toFixed(1),
    f1_score: (h.f1_score * 100).toFixed(1)
  }))

  return (
    <div className="glass-panel" style={{ background: 'transparent', border: 'none', boxShadow: 'none', display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      <div className="orbital-container">
        
        {/* Core Global Metric */}
        <div className="core-metric">
          <div className="core-value">
            {(roc_auc * 100).toFixed(1)}<span style={{ fontSize: '2rem', color: 'var(--text-secondary)' }}>%</span>
          </div>
          <div className="core-label">Global ROC AUC</div>
        </div>

        {/* Satellite Client/Round Metrics */}
        <div className="satellite-grid">
          <div className="satellite-card">
            <div className="sat-label">Current Training Round</div>
            <div className="sat-value">{data.current_round}</div>
          </div>
          <div className="satellite-card">
            <div className="sat-label">Active Clients</div>
            <div className="sat-value">{data.active_clients}</div>
          </div>
          <div className="satellite-card">
            <div className="sat-label">Global F1 Score</div>
            <div className="sat-value">{(f1_score * 100).toFixed(1)}%</div>
          </div>
          <div className="satellite-card">
            <div className="sat-label">Global Precision</div>
            <div className="sat-value">{(precision * 100).toFixed(1)}%</div>
          </div>
        </div>
      </div>

      <div className="glass-panel" style={{ height: '300px', padding: '2rem', marginTop: '1rem' }}>
        <h3 style={{ marginBottom: '1.5rem', fontSize: '1.2rem', fontWeight: 500 }}>Historical Assessment Trends</h3>
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={chartData}>
            <defs>
              <linearGradient id="colorRoc" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="var(--accent-base)" stopOpacity={0.8}/>
                <stop offset="95%" stopColor="var(--accent-base)" stopOpacity={0}/>
              </linearGradient>
            </defs>
            <XAxis dataKey="round" stroke="var(--text-secondary)" tick={{fill: 'var(--text-secondary)'}} />
            <YAxis domain={['auto', 'auto']} stroke="var(--text-secondary)" tick={{fill: 'var(--text-secondary)'}} />
            <Tooltip contentStyle={{ backgroundColor: 'rgba(10, 10, 15, 0.9)', border: '1px solid var(--glass-border)', borderRadius: '12px' }} />
            <Area type="monotone" dataKey="roc_auc" stroke="var(--accent-base)" strokeWidth={3} fillOpacity={1} fill="url(#colorRoc)" />
          </AreaChart>
        </ResponsiveContainer>
      </div>

    </div>
  )
}

export default MetricsOverview
