import { useState, useEffect } from 'react'
import axios from 'axios'
import MetricsOverview from './components/MetricsOverview'
import ShapExplainability from './components/ShapExplainability'

const API_URL = 'http://localhost:8000/api'

function App() {
  const [activeTab, setActiveTab] = useState('metrics')
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)

  const fetchData = async () => {
    try {
      const res = await axios.get(`${API_URL}/metrics`)
      setData(res.data)
      setLoading(false)
    } catch (err) {
      console.error(err)
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchData()
    const interval = setInterval(fetchData, 5000)
    return () => clearInterval(interval)
  }, [])

  if (loading) return <div className="viewport-container">Loading...</div>

  return (
    <>
      <div className="ambient-bg"></div>
      
      <div className="viewport-container">
        <header className="floating-header">
          <div className="brand-pill">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{color: 'var(--accent-base)'}}>
              <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
            </svg>
            Transaction Monitor
          </div>

          <div className="system-status">
            <div className="status-indicator">
              <div className="pulse-dot"></div>
              Network Status: Healthy
            </div>
            <div className="status-indicator">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect><path d="M7 11V7a5 5 0 0 1 10 0v4"></path></svg>
              Data Privacy: Local
            </div>
          </div>
        </header>

        <main className="display-stage">
          {activeTab === 'metrics' && <MetricsOverview data={data} />}
          {activeTab === 'shap' && <ShapExplainability apiUrl={API_URL} clients={['client_1', 'client_2', 'client_3']} />}
        </main>

        <nav className="mac-dock">
          <button 
            className={`dock-item ${activeTab === 'metrics' ? 'active' : ''}`}
            onClick={() => setActiveTab('metrics')}
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10"></circle>
              <line x1="12" y1="8" x2="12" y2="16"></line>
              <line x1="8" y1="12" x2="16" y2="12"></line>
            </svg>
            Global View
          </button>
          
          <button 
            className={`dock-item ${activeTab === 'shap' ? 'active' : ''}`}
            onClick={() => setActiveTab('shap')}
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="18" y1="20" x2="18" y2="10"></line>
              <line x1="12" y1="20" x2="12" y2="4"></line>
              <line x1="6" y1="20" x2="6" y2="14"></line>
            </svg>
            Feature Analysis
          </button>
        </nav>
      </div>
    </>
  )
}

export default App
