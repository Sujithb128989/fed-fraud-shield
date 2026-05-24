import React, { useState, useEffect } from 'react'
import axios from 'axios'

const ShapExplainability = ({ apiUrl, clients }) => {
  const [selectedClient, setSelectedClient] = useState('')
  const [shapData, setShapData] = useState([])
  const [loading, setLoading] = useState(false)

  const uniqueClients = [...new Set(clients)]

  useEffect(() => {
    if (uniqueClients.length > 0 && !selectedClient) {
      setSelectedClient(uniqueClients[0])
    }
  }, [uniqueClients, selectedClient])

  useEffect(() => {
    const fetchShap = async () => {
      if (!selectedClient) return
      setLoading(true)
      try {
        const res = await axios.get(`${apiUrl}/shap/${selectedClient}`)
        setShapData(res.data)
      } catch (err) {
        console.error(err)
        setShapData([])
      }
      setLoading(false)
    }
    fetchShap()
  }, [selectedClient, apiUrl])

  // Find max absolute value to scale bars properly
  const maxVal = Math.max(...shapData.map(d => Math.abs(d.value)), 0.001)

  return (
    <div className="glass-panel" style={{ maxWidth: '800px', width: '100%' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
        <h3 style={{ fontSize: '1.25rem', fontWeight: 500 }}>Anomaly Drivers (Local)</h3>
        
        {uniqueClients.length > 0 && (
          <select 
            className="glass-select"
            value={selectedClient} 
            onChange={(e) => setSelectedClient(e.target.value)}
          >
            {uniqueClients.map(c => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
        )}
      </div>

      {loading ? (
        <div style={{ color: 'var(--text-secondary)' }}>Decrypting driver analysis...</div>
      ) : shapData.length === 0 ? (
        <div style={{ color: 'var(--text-secondary)' }}>No driver analysis data found.</div>
      ) : (
        <div className="waterfall-container">
          {shapData.map((d, i) => {
            const pct = (Math.abs(d.value) / maxVal) * 100;
            return (
              <div className="waterfall-item" key={i}>
                <div className="wf-feature">{d.feature}</div>
                <div className="wf-bar-track">
                  <div className="wf-bar-fill" style={{ width: `${pct}%`, animationDelay: `${i * 0.05}s` }}></div>
                </div>
                <div className="wf-value">{d.value > 0 ? '+' : ''}{d.value.toFixed(4)}</div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

export default ShapExplainability
