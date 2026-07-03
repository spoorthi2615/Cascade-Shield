import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import './App.css';

const API_BASE = 'http://localhost:8000/api/dashboard';

// ---------- thermal color scale ----------
function heatColor(p) {
  const stops = [
    { t: 0, c: [30, 41, 59] },
    { t: 0.35, c: [45, 212, 191] },
    { t: 0.65, c: [251, 191, 36] },
    { t: 1, c: [239, 68, 68] },
  ];
  let a = stops[0], b = stops[stops.length - 1];
  for (let k = 0; k < stops.length - 1; k++) {
    if (p >= stops[k].t && p <= stops[k + 1].t) { a = stops[k]; b = stops[k + 1]; break; }
  }
  const span = b.t - a.t || 1;
  const f = (p - a.t) / span;
  const c = a.c.map((v, idx) => Math.round(v + (b.c[idx] - v) * f));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

// ---------- ROC-AUC, rank-sum method (tie-aware, exact) ----------
function computeAUC(probs, truth) {
  const nPos = truth.filter((t) => t === 1).length;
  const nNeg = truth.length - nPos;
  if (!nPos || !nNeg) return null;
  const rows = probs.map((p, i) => ({ p, t: truth[i] })).sort((a, b) => a.p - b.p);
  let rank = 1, rankSumPos = 0, i = 0;
  while (i < rows.length) {
    let j = i;
    while (j + 1 < rows.length && rows[j + 1].p === rows[i].p) j++;
    const avgRank = (rank + rank + (j - i)) / 2;
    for (let k = i; k <= j; k++) if (rows[k].t) rankSumPos += avgRank;
    rank += j - i + 1;
    i = j + 1;
  }
  return (rankSumPos - (nPos * (nPos + 1)) / 2) / (nPos * nNeg);
}

function buildHistogram(probs, truth, bins = 10) {
  const infectedCounts = new Array(bins).fill(0);
  const safeCounts = new Array(bins).fill(0);
  probs.forEach((p, i) => {
    const b = Math.min(bins - 1, Math.floor(p * bins));
    if (truth[i]) infectedCounts[b]++; else safeCounts[b]++;
  });
  return { infectedCounts, safeCounts, max: Math.max(1, ...infectedCounts, ...safeCounts) };
}

const MODES = [
  { key: 'ground_truth', label: 'GROUND TRUTH', accent: '#FB7185' },
  { key: 'gnn', label: 'CHAMPION GNN', accent: '#38BDF8' },
  { key: 'seir', label: 'CLASSICAL SEIR', accent: '#A78BFA' },
];

function App() {
  const [scenarios, setScenarios] = useState([]);
  const [selectedScenarioId, setSelectedScenarioId] = useState(null);
  const [scenarioData, setScenarioData] = useState(null);
  const [predictionData, setPredictionData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [viewMode, setViewMode] = useState('ground_truth');

  // scenario combobox state (scales to 100+ scenarios)
  const [scenarioFilter, setScenarioFilter] = useState('');
  const [dropdownOpen, setDropdownOpen] = useState(false);

  const fgRef = useRef();

  useEffect(() => {
    fetch(`${API_BASE}/scenarios`)
      .then((res) => res.json())
      .then((data) => {
        setScenarios(data.scenarios);
        if (data.scenarios.length > 0) setSelectedScenarioId(data.scenarios[0].id);
      })
      .catch((err) => console.error('Error loading scenarios:', err));
  }, []);

  useEffect(() => {
    if (selectedScenarioId === null) return;
    setLoading(true);
    Promise.all([
      fetch(`${API_BASE}/scenario/${selectedScenarioId}`).then((res) => res.json()),
      fetch(`${API_BASE}/predict/${selectedScenarioId}`).then((res) => res.json()),
    ])
      .then(([graphData, preds]) => {
        setScenarioData(graphData);
        setPredictionData(preds);
        setLoading(false);
        if (fgRef.current) setTimeout(() => fgRef.current.zoomToFit(400, 60), 500);
      })
      .catch((err) => {
        console.error('Error loading scenario details:', err);
        setLoading(false);
      });
  }, [selectedScenarioId]);

  const nodeColorFor = useCallback(
    (node) => {
      if (node.is_origin) return '#FBBF24';
      if (viewMode === 'ground_truth') return heatColor(node.is_infected_gt ? 1 : 0.08);
      if (!predictionData) return heatColor(0);
      const probs = viewMode === 'gnn' ? predictionData.gnn_probs : predictionData.seir_probs;
      return heatColor(probs[node.id]);
    },
    [viewMode, predictionData]
  );

  const paintNode = useCallback(
    (node, ctx, globalScale) => {
      const baseR = node.is_origin ? 7 : 4.3;
      if (node.is_origin) {
        // NOTE: relies on force-graph's continuous canvas render loop for the pulse.
        // If the ring doesn't animate once physics settles in your build, fall back to
        // ticking a small dummy state via setInterval(() => setTick(t => t+1), 50)
        // to force re-renders.
        const pulse = (Math.sin(Date.now() / 300) + 1) / 2;
        const ringR = baseR + 4 + pulse * 8;
        ctx.beginPath();
        ctx.arc(node.x, node.y, ringR, 0, 2 * Math.PI);
        ctx.strokeStyle = `rgba(251, 191, 36, ${0.6 * (1 - pulse)})`;
        ctx.lineWidth = 1.5 / globalScale;
        ctx.stroke();
      }
      ctx.beginPath();
      ctx.arc(node.x, node.y, baseR, 0, 2 * Math.PI);
      ctx.fillStyle = nodeColorFor(node);
      ctx.fill();
      ctx.lineWidth = 1 / globalScale;
      ctx.strokeStyle = 'rgba(8,13,22,0.85)';
      ctx.stroke();
    },
    [nodeColorFor]
  );

  const paintPointerArea = useCallback((node, color, ctx) => {
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(node.x, node.y, (node.is_origin ? 7 : 4.3) + 2, 0, 2 * Math.PI);
    ctx.fill();
  }, []);

  const currentProbs =
    viewMode === 'gnn' ? predictionData?.gnn_probs : viewMode === 'seir' ? predictionData?.seir_probs : null;

  const auc = useMemo(() => {
    if (!currentProbs || !predictionData) return null;
    return computeAUC(currentProbs, predictionData.ground_truth);
  }, [currentProbs, predictionData]);

  const histogram = useMemo(() => {
    if (!currentProbs || !predictionData) return null;
    return buildHistogram(currentProbs, predictionData.ground_truth);
  }, [currentProbs, predictionData]);

  const activeMode = MODES.find((m) => m.key === viewMode);

  const filteredScenarios = useMemo(() => {
    if (!scenarioFilter) return scenarios;
    return scenarios.filter(
      (s) => String(s.id).includes(scenarioFilter) || String(s.cascade_size).includes(scenarioFilter)
    );
  }, [scenarios, scenarioFilter]);

  const selectedScenario = scenarios.find((s) => s.id === selectedScenarioId);

  return (
    <div className="dashboard-container">
      <header className="dashboard-header">
        <div className="title-block">
          <div className="title">CASCADESHIELD</div>
          <div className="subtitle">INFRASTRUCTURE CASCADE FAILURE PREDICTION</div>
        </div>

        <div className="status-pill">
          <span className="status-dot" />
          SYSTEM NOMINAL
        </div>

        {selectedScenario && (
          <div className="readout">
            <div className="readout-label">SCENARIO {String(selectedScenario.id).padStart(3, '0')} · CASCADE SIZE</div>
            <div className="readout-value">K = {selectedScenario.cascade_size}</div>
          </div>
        )}
      </header>

      <main className="dashboard-main">
        <div className="graph-container">
          {loading ? (
            <div className="loading">Loading...</div>
          ) : (
            <ForceGraph2D
              ref={fgRef}
              graphData={scenarioData || { nodes: [], links: [] }}
              nodeCanvasObject={paintNode}
              nodeCanvasObjectMode={() => 'replace'}
              nodePointerAreaPaint={paintPointerArea}
              nodeLabel={(node) => `
                <div style="font-family:'JetBrains Mono',monospace;font-size:11px;line-height:1.6;padding:4px 2px;color:#E7ECF3;">
                  <strong>NODE ${String(node.id).padStart(3, '0')}</strong>${node.is_origin ? ' · ORIGIN' : ''}<br/>
                  GT: ${node.is_infected_gt ? 'INFECTED' : 'SAFE'}
                  ${!node.is_origin && predictionData ? `<br/>GNN ${predictionData.gnn_probs[node.id].toFixed(3)}<br/>SEIR ${predictionData.seir_probs[node.id].toFixed(3)}` : ''}
                </div>
              `}
              linkColor={() => 'rgba(56,189,248,0.2)'}
              linkWidth={1}
              linkDirectionalParticles={1}
              linkDirectionalParticleSpeed={0.005}
              linkDirectionalParticleColor={() => '#38BDF8'}
              backgroundColor="#080D16"
            />
          )}
          <div className="scope-caption">VIEW · {activeMode.label}</div>
        </div>

        <aside className="metrics-panel">
          <div className="panel-section">
            <div className="section-label">CONTROLS</div>
            
            <div className="scenario-combobox" style={{ width: '100%', marginBottom: '16px' }}>
              <input
                type="text"
                placeholder={selectedScenario ? `Scenario ${selectedScenario.id} (K=${selectedScenario.cascade_size})` : 'Search scenario...'}
                value={scenarioFilter}
                onChange={(e) => { setScenarioFilter(e.target.value); setDropdownOpen(true); }}
                onFocus={() => setDropdownOpen(true)}
                onBlur={() => setTimeout(() => setDropdownOpen(false), 150)}
              />
              {dropdownOpen && (
                <div className="scenario-dropdown">
                  {filteredScenarios.slice(0, 200).map((s) => (
                    <div
                      key={s.id}
                      className={`scenario-option ${s.id === selectedScenarioId ? 'active' : ''}`}
                      onMouseDown={() => {
                        setSelectedScenarioId(s.id);
                        setScenarioFilter('');
                        setDropdownOpen(false);
                      }}
                    >
                      <span>Scenario {String(s.id).padStart(3, '0')}</span>
                      <span className="k-badge">K={s.cascade_size}</span>
                    </div>
                  ))}
                  {filteredScenarios.length === 0 && <div className="scenario-option empty">No matches</div>}
                </div>
              )}
            </div>

            <div className="view-toggles" style={{ flexDirection: 'column', gap: '8px' }}>
              {MODES.map((m) => (
                <button
                  key={m.key}
                  className={viewMode === m.key ? 'active' : ''}
                  style={{ '--mode-accent': m.accent, width: '100%', textAlign: 'left' }}
                  onClick={() => setViewMode(m.key)}
                >
                  {m.label}
                </button>
              ))}
            </div>
          </div>
          <div className="panel-section">
            <div className="section-label">METRICS</div>
            <div className="gauge-grid">
              <div className="gauge">
                <div className="gauge-label">NODES</div>
                <div className="gauge-value">{scenarioData?.nodes.length ?? '—'}</div>
              </div>
              <div className="gauge">
                <div className="gauge-label">EDGES</div>
                <div className="gauge-value">{scenarioData?.links.length ?? '—'}</div>
              </div>
              <div className="gauge">
                <div className="gauge-label">CASCADE K</div>
                <div className="gauge-value">{predictionData?.K ?? '—'}</div>
              </div>
              <div className="gauge">
                <div className="gauge-label">ROC-AUC</div>
                <div className="gauge-value accent" style={{ '--accent': activeMode.accent }}>
                  {auc !== null ? auc.toFixed(3) : '—'}
                </div>
              </div>
            </div>
          </div>

          <div className="panel-section">
            <div className="section-label">SCORE SEPARATION</div>
            {histogram ? (
              <>
                <div className="hist-wrap">
                  <div className="hist-row">
                    {histogram.infectedCounts.map((c, i) => (
                      <div key={i} className="hist-bar" style={{ height: `${(c / histogram.max) * 100}%`, background: '#EF4444', opacity: 0.75 }} />
                    ))}
                  </div>
                  <div className="hist-mid" />
                  <div className="hist-row bottom">
                    {histogram.safeCounts.map((c, i) => (
                      <div key={i} className="hist-bar" style={{ height: `${(c / histogram.max) * 100}%`, background: '#2DD4BF', opacity: 0.6 }} />
                    ))}
                  </div>
                </div>
                <div className="hist-legend">
                  <span><i style={{ background: '#EF4444' }} /> INFECTED</span>
                  <span><i style={{ background: '#2DD4BF' }} /> SAFE</span>
                </div>
              </>
            ) : (
              <div className="hist-caption">Switch to GNN or SEIR to see class separation.</div>
            )}
          </div>

          <div className="panel-section" style={{ borderBottom: 'none' }}>
            <div className="section-label">LEGEND</div>
            <div className="legend-scale" />
            <div className="legend-labels"><span>0.0 SAFE</span><span>1.0 CRITICAL</span></div>
            <div className="legend-origin"><i /> ORIGIN NODE</div>
          </div>
        </aside>
      </main>
    </div>
  );
}

export default App;
