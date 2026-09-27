import React, { useState, useEffect, useRef } from 'react';
import {
  Satellite,
  Compass,
  Layers,
  Activity,
  CheckCircle2,
  AlertCircle,
  Upload,
  ArrowRight,
  Maximize2,
  Eye,
  Sliders,
  Sparkles,
  HelpCircle,
  Cpu,
  RefreshCw,
  FolderOpen,
  SplitSquareVertical,
  ChevronRight,
  ShieldAlert,
  Info
} from 'lucide-react';

const API_BASE = 'http://127.0.0.1:8000';

export default function App() {
  // Navigation & Mode
  const [activeMode, setActiveMode] = useState('AUTO'); // AUTO, VQA, GROUNDING, CHANGE_DETECTION, FUSION
  const [backendHealth, setBackendHealth] = useState({ status: 'checking', specialists: [] });
  const [sampleScenarios, setSampleScenarios] = useState([]);
  const [activeScenario, setActiveScenario] = useState(null);

  // Inputs
  const [query, setQuery] = useState('What is the dominant land cover in this image?');
  const [imageA, setImageA] = useState(null);
  const [imageB, setImageB] = useState(null);
  const [previewA, setPreviewA] = useState('/sample_data/sc1_vqa_agricultural_lake.png');
  const [previewB, setPreviewB] = useState(null);

  // Viewer State
  const [viewerTab, setViewerTab] = useState('overlay'); // 'original', 'overlay', 'split'
  const [overlayOpacity, setOverlayOpacity] = useState(0.85);
  const [splitPos, setSplitPos] = useState(50);
  const [zoomLevel, setZoomLevel] = useState(1);

  // Analysis State
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [showDocsModal, setShowDocsModal] = useState(false);

  const fileInputARef = useRef(null);
  const fileInputBRef = useRef(null);

  // Check backend health & fetch sample scenarios on mount
  useEffect(() => {
    fetch(`${API_BASE}/api/health`)
      .then((res) => res.json())
      .then((data) => setBackendHealth(data))
      .catch((err) => {
        console.warn('Backend not responding, using offline fallback info:', err);
        setBackendHealth({ status: 'offline', error: 'Connecting to backend...' });
      });

    fetch(`${API_BASE}/api/samples`)
      .then((res) => res.json())
      .then((data) => {
        if (Array.isArray(data) && data.length > 0) {
          setSampleScenarios(data);
          setActiveScenario(data[0]);
        }
      })
      .catch((err) => {
        // Fallback to local public samples catalog
        fetch('/sample_data/sample_scenarios.json')
          .then((res) => res.json())
          .then((data) => {
            setSampleScenarios(data);
            setActiveScenario(data[0]);
          })
          .catch((e) => console.error('Failed to load sample scenarios', e));
      });
  }, []);

  // Quick-load demo preset scenario
  const handleSelectScenario = async (sc) => {
    setActiveScenario(sc);
    setError(null);
    setQuery(sc.default_query);

    // Set preview paths
    const pA = `/sample_data/${sc.primary_image}`;
    setPreviewA(pA);
    setImageA(null); // Will convert from path on analyze if file not uploaded

    if (sc.secondary_image) {
      const pB = `/sample_data/${sc.secondary_image}`;
      setPreviewB(pB);
      setImageB(null);
    } else {
      setPreviewB(null);
      setImageB(null);
    }

    if (sc.task === 'CHANGE_DETECTION') {
      setActiveMode('CHANGE_DETECTION');
      setViewerTab('overlay');
    } else if (sc.task === 'FUSION') {
      setActiveMode('FUSION');
      setViewerTab('overlay');
    } else if (sc.task === 'GROUNDING') {
      setActiveMode('GROUNDING');
      setViewerTab('overlay');
    } else {
      setActiveMode('VQA');
      setViewerTab('overlay');
    }

    // Trigger analysis immediately for responsive live demonstration
    runAnalysisWithUrls(sc.default_query, sc.task, pA, sc.secondary_image ? `/sample_data/${sc.secondary_image}` : null);
  };

  // Convert image URL to Blob/File object for form upload
  const urlToFile = async (url, filename) => {
    const res = await fetch(url);
    const blob = await res.blob();
    return new File([blob], filename, { type: blob.type || 'image/png' });
  };

  const runAnalysisWithUrls = async (customQuery, taskOverride, urlA, urlB) => {
    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('query', customQuery || query);
      if (taskOverride && taskOverride !== 'AUTO') {
        formData.append('task', taskOverride);
      }

      // Append primary image
      if (imageA) {
        formData.append('images', imageA);
      } else if (urlA) {
        const fA = await urlToFile(urlA, 'primary_sample.png');
        formData.append('images', fA);
      }

      // Append secondary image if needed
      if (imageB) {
        formData.append('images', imageB);
      } else if (urlB) {
        const fB = await urlToFile(urlB, 'secondary_sample.png');
        formData.append('images', fB);
      }

      const res = await fetch(`${API_BASE}/api/query`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Server returned ${res.status}`);
      }

      const data = await res.json();
      setResult(data);
      setViewerTab('overlay');
    } catch (err) {
      console.error('Analysis error:', err);
      setError(err.message || 'Failed to analyze satellite imagery. Please check backend connection.');
    } finally {
      setLoading(false);
    }
  };

  const handleManualAnalyze = () => {
    runAnalysisWithUrls(
      query,
      activeMode === 'AUTO' ? null : activeMode,
      previewA,
      previewB
    );
  };

  const handleFileUploadA = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImageA(file);
      setPreviewA(URL.createObjectURL(file));
      setResult(null);
    }
  };

  const handleFileUploadB = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImageB(file);
      setPreviewB(URL.createObjectURL(file));
      setResult(null);
    }
  };

  const isDualMode =
    activeMode === 'CHANGE_DETECTION' ||
    activeMode === 'FUSION' ||
    previewB !== null ||
    (activeScenario && activeScenario.image_count === 2);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', overflow: 'hidden' }}>
      {/* 1. TOP HEADER */}
      <header
        style={{
          background: 'rgba(15, 23, 42, 0.95)',
          borderBottom: '1px solid rgba(51, 65, 85, 0.7)',
          padding: '12px 24px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          zIndex: 40,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <div
            style={{
              width: '42px',
              height: '42px',
              borderRadius: '10px',
              background: 'linear-gradient(135deg, #0ea5e9, #10b981)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: '0 0 15px rgba(14, 165, 233, 0.4)',
            }}
          >
            <Satellite size={24} color="#ffffff" />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <h1 style={{ fontSize: '18px', fontWeight: '700', letterSpacing: '0.04em', color: '#f8fafc' }}>
                SATQUERY AI
              </h1>
              <span
                style={{
                  fontSize: '11px',
                  fontWeight: '600',
                  padding: '2px 8px',
                  borderRadius: '12px',
                  background: 'rgba(16, 185, 129, 0.15)',
                  color: '#34d399',
                  border: '1px solid rgba(16, 185, 129, 0.3)',
                }}
              >
                SIH 2026 / ISRO-SAC • PS-26167
              </span>
            </div>
            <p style={{ fontSize: '12px', color: '#94a3b8' }}>
              Conversational Satellite Intelligence & Remote Sensing Vision-Language Engine
            </p>
          </div>
        </div>

        {/* Center: Benchmarks loaded */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ fontSize: '11px', color: '#64748b', fontWeight: '500' }}>BENCHMARKS:</span>
          {['BigEarthNet.txt', 'VRSBench', 'RSVQA', 'CDVQA'].map((bm) => (
            <span
              key={bm}
              style={{
                fontSize: '11px',
                padding: '3px 8px',
                borderRadius: '6px',
                background: 'rgba(30, 41, 59, 0.8)',
                color: '#94a3b8',
                border: '1px solid rgba(51, 65, 85, 0.6)',
              }}
            >
              {bm}
            </span>
          ))}
        </div>

        {/* Right Status Indicator */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              padding: '6px 12px',
              borderRadius: '20px',
              background: backendHealth.status === 'ok' ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)',
              border: `1px solid ${backendHealth.status === 'ok' ? 'rgba(16, 185, 129, 0.3)' : 'rgba(239, 68, 68, 0.3)'}`,
            }}
          >
            <div
              style={{
                width: '8px',
                height: '8px',
                borderRadius: '50%',
                background: backendHealth.status === 'ok' ? '#10b981' : '#ef4444',
                boxShadow: `0 0 8px ${backendHealth.status === 'ok' ? '#10b981' : '#ef4444'}`,
              }}
            />
            <span style={{ fontSize: '12px', fontWeight: '500', color: backendHealth.status === 'ok' ? '#34d399' : '#f87171' }}>
              {backendHealth.status === 'ok' ? 'Engine Ready' : 'Connecting...'}
            </span>
          </div>

          <button
            onClick={() => setShowDocsModal(true)}
            style={{
              background: 'transparent',
              border: '1px solid #334155',
              borderRadius: '8px',
              padding: '6px 10px',
              color: '#94a3b8',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              fontSize: '12px',
            }}
          >
            <Info size={14} />
            <span>SIH Specs</span>
          </button>
        </div>
      </header>

      {/* 2. MAIN LAYOUT: SIDEBAR + WORKSPACE */}
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        {/* SIDEBAR: Specialist Navigation & Presets */}
        <aside
          style={{
            width: '280px',
            background: 'rgba(15, 23, 42, 0.85)',
            borderRight: '1px solid rgba(51, 65, 85, 0.6)',
            display: 'flex',
            flexDirection: 'column',
            overflowY: 'auto',
            padding: '16px 12px',
          }}
        >
          {/* Mode Selection */}
          <div style={{ marginBottom: '20px' }}>
            <h3 style={{ fontSize: '11px', fontWeight: '700', color: '#64748b', textTransform: 'uppercase', marginBottom: '10px', letterSpacing: '0.05em' }}>
              Analysis Specialist
            </h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
              {[
                { id: 'AUTO', label: 'AI Router (Autonomous)', icon: Sparkles, color: '#38bdf8' },
                { id: 'VQA', label: 'Visual QA (VQA)', icon: HelpCircle, color: '#a855f7' },
                { id: 'GROUNDING', label: 'Visual Grounding', icon: Compass, color: '#10b981' },
                { id: 'CHANGE_DETECTION', label: 'Change Detection', icon: SplitSquareVertical, color: '#f59e0b' },
                { id: 'FUSION', label: 'Optical + SAR Fusion', icon: Layers, color: '#06b6d4' },
              ].map((m) => {
                const Icon = m.icon;
                const isSelected = activeMode === m.id;
                return (
                  <button
                    key={m.id}
                    onClick={() => setActiveMode(m.id)}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '10px',
                      padding: '10px 12px',
                      borderRadius: '8px',
                      background: isSelected ? 'rgba(30, 41, 59, 0.9)' : 'transparent',
                      border: isSelected ? `1px solid ${m.color}` : '1px solid transparent',
                      color: isSelected ? '#f8fafc' : '#94a3b8',
                      cursor: 'pointer',
                      fontSize: '13px',
                      fontWeight: isSelected ? '600' : '400',
                      textAlign: 'left',
                      transition: 'all 0.15s ease',
                    }}
                  >
                    <Icon size={16} color={isSelected ? m.color : '#64748b'} />
                    <span>{m.label}</span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Preset Demo Scenarios */}
          <div style={{ flex: 1 }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '10px' }}>
              <h3 style={{ fontSize: '11px', fontWeight: '700', color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                SIH Demo Scenarios
              </h3>
              <span style={{ fontSize: '10px', color: '#0ea5e9', fontWeight: '600' }}>1-CLICK LOAD</span>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {sampleScenarios.map((sc) => {
                const isCurrent = activeScenario?.id === sc.id;
                return (
                  <div
                    key={sc.id}
                    onClick={() => handleSelectScenario(sc)}
                    style={{
                      padding: '10px',
                      borderRadius: '8px',
                      background: isCurrent ? 'rgba(30, 41, 59, 0.95)' : 'rgba(22, 32, 50, 0.5)',
                      border: isCurrent ? '1px solid #0ea5e9' : '1px solid rgba(51, 65, 85, 0.4)',
                      cursor: 'pointer',
                      transition: 'all 0.15s ease',
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '4px' }}>
                      <span
                        style={{
                          fontSize: '10px',
                          fontWeight: '700',
                          padding: '1px 6px',
                          borderRadius: '4px',
                          background:
                            sc.task === 'VQA'
                              ? 'rgba(168, 85, 247, 0.2)'
                              : sc.task === 'GROUNDING'
                              ? 'rgba(16, 185, 129, 0.2)'
                              : sc.task === 'CHANGE_DETECTION'
                              ? 'rgba(245, 158, 11, 0.2)'
                              : 'rgba(6, 182, 212, 0.2)',
                          color:
                            sc.task === 'VQA'
                              ? '#c084fc'
                              : sc.task === 'GROUNDING'
                              ? '#34d399'
                              : sc.task === 'CHANGE_DETECTION'
                              ? '#fbbf24'
                              : '#22d3ee',
                        }}
                      >
                        {sc.task}
                      </span>
                      <span style={{ fontSize: '10px', color: '#64748b' }}>
                        {sc.image_count === 2 ? '2 Sensors/Dates' : 'Single Scene'}
                      </span>
                    </div>
                    <div style={{ fontSize: '12px', fontWeight: '600', color: '#f1f5f9', marginBottom: '4px' }}>
                      {sc.title}
                    </div>
                    <div style={{ fontSize: '11px', color: '#94a3b8', lineHeight: '1.3' }}>
                      {sc.benchmark}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* ISRO SAC Notice Footer */}
          <div style={{ marginTop: '16px', padding: '10px', borderRadius: '8px', background: 'rgba(15, 23, 42, 0.6)', border: '1px solid rgba(51, 65, 85, 0.4)' }}>
            <div style={{ fontSize: '11px', fontWeight: '600', color: '#94a3b8', marginBottom: '4px' }}>
              ISRO SAC Evaluation Ready
            </div>
            <div style={{ fontSize: '10px', color: '#64748b', lineHeight: '1.4' }}>
              Accepts Cartosat-2S Optical + RISAT SAR pre-registered pairs and Sentinel-1/2 benchmark datasets.
            </div>
          </div>
        </aside>

        {/* WORKSPACE & RESULTS */}
        <main style={{ flex: 1, display: 'flex', flexDirection: 'column', overflowY: 'auto', padding: '16px', gap: '16px' }}>
          {/* A. WORKSPACE TOP BAR: UPLOAD DUAL SLOTS + QUERY INPUT */}
          <div className="glass-panel" style={{ padding: '16px' }}>
            <div style={{ display: 'grid', gridTemplateColumns: isDualMode ? '1fr 1fr' : '1fr', gap: '12px', marginBottom: '14px' }}>
              {/* Slot A: Primary / Baseline / Optical */}
              <div
                style={{
                  border: '1px dashed #334155',
                  borderRadius: '8px',
                  padding: '10px 14px',
                  background: 'rgba(15, 23, 42, 0.6)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                  {previewA ? (
                    <img
                      src={previewA}
                      alt="Primary Slot"
                      style={{ width: '48px', height: '48px', borderRadius: '6px', objectFit: 'cover', border: '1px solid #334155' }}
                    />
                  ) : (
                    <div style={{ width: '48px', height: '48px', borderRadius: '6px', background: '#1e293b', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                      <Upload size={20} color="#64748b" />
                    </div>
                  )}
                  <div>
                    <div style={{ fontSize: '12px', fontWeight: '600', color: '#f1f5f9' }}>
                      {isDualMode ? 'Image 1: Baseline (T1) / Optical Acquisition' : 'Input Satellite Image'}
                    </div>
                    <div style={{ fontSize: '11px', color: '#64748b' }}>
                      {imageA ? imageA.name : (previewA ? previewA.split('/').pop() : 'Click upload or drag tile')}
                    </div>
                  </div>
                </div>

                <div style={{ display: 'flex', gap: '6px' }}>
                  <input type="file" ref={fileInputARef} onChange={handleFileUploadA} accept="image/*" style={{ display: 'none' }} />
                  <button
                    onClick={() => fileInputARef.current?.click()}
                    style={{
                      background: '#1e293b',
                      border: '1px solid #334155',
                      borderRadius: '6px',
                      padding: '6px 12px',
                      fontSize: '11px',
                      color: '#cbd5e1',
                      cursor: 'pointer',
                    }}
                  >
                    Change Image
                  </button>
                </div>
              </div>

              {/* Slot B: Temporal (T2) / SAR Radar */}
              {isDualMode && (
                <div
                  style={{
                    border: '1px dashed #334155',
                    borderRadius: '8px',
                    padding: '10px 14px',
                    background: 'rgba(15, 23, 42, 0.6)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                    {previewB ? (
                      <img
                        src={previewB}
                        alt="Secondary Slot"
                        style={{ width: '48px', height: '48px', borderRadius: '6px', objectFit: 'cover', border: '1px solid #334155' }}
                      />
                    ) : (
                      <div style={{ width: '48px', height: '48px', borderRadius: '6px', background: '#1e293b', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                        <Upload size={20} color="#64748b" />
                      </div>
                    )}
                    <div>
                      <div style={{ fontSize: '12px', fontWeight: '600', color: '#f1f5f9' }}>
                        {activeMode === 'FUSION' ? 'Image 2: SAR Radar (Sentinel-1 / RISAT)' : 'Image 2: Current (T2) / Post-Event'}
                      </div>
                      <div style={{ fontSize: '11px', color: '#64748b' }}>
                        {imageB ? imageB.name : (previewB ? previewB.split('/').pop() : 'Upload co-registered image')}
                      </div>
                    </div>
                  </div>

                  <div style={{ display: 'flex', gap: '6px' }}>
                    <input type="file" ref={fileInputBRef} onChange={handleFileUploadB} accept="image/*" style={{ display: 'none' }} />
                    <button
                      onClick={() => fileInputBRef.current?.click()}
                      style={{
                        background: '#1e293b',
                        border: '1px solid #334155',
                        borderRadius: '6px',
                        padding: '6px 12px',
                        fontSize: '11px',
                        color: '#cbd5e1',
                        cursor: 'pointer',
                      }}
                    >
                      Change Image
                    </button>
                  </div>
                </div>
              )}
            </div>

            {/* Query Input Box */}
            <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
              <div style={{ flex: 1, position: 'relative' }}>
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleManualAnalyze()}
                  placeholder="Ask any natural-language question about this satellite scene..."
                  style={{
                    width: '100%',
                    padding: '14px 16px',
                    borderRadius: '8px',
                    background: 'rgba(15, 23, 42, 0.95)',
                    border: '1px solid #334155',
                    color: '#f8fafc',
                    fontSize: '14px',
                    outline: 'none',
                    transition: 'border 0.2s',
                  }}
                />
              </div>

              <button
                onClick={handleManualAnalyze}
                disabled={loading}
                style={{
                  background: 'linear-gradient(135deg, #0284c7, #059669)',
                  border: 'none',
                  borderRadius: '8px',
                  padding: '14px 24px',
                  color: '#ffffff',
                  fontSize: '14px',
                  fontWeight: '600',
                  cursor: loading ? 'not-allowed' : 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  boxShadow: '0 0 15px rgba(2, 132, 199, 0.3)',
                }}
              >
                {loading ? <RefreshCw size={18} className="animate-spin" /> : <Sparkles size={18} />}
                <span>{loading ? 'Analyzing...' : 'Run Query'}</span>
              </button>
            </div>

            {/* Suggested Query Chips */}
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', marginTop: '10px' }}>
              <span style={{ fontSize: '11px', color: '#64748b', alignSelf: 'center', marginRight: '4px' }}>Quick Prompts:</span>
              {(activeScenario?.suggested_queries || [
                'What is the dominant land cover in this image?',
                'Show me the water bodies.',
                'Find the built-up regions.',
                'What changed between these two images?',
                'Combine optical and SAR imagery to identify built-up areas.',
              ]).map((prompt) => (
                <button
                  key={prompt}
                  onClick={() => {
                    setQuery(prompt);
                    runAnalysisWithUrls(prompt, activeMode === 'AUTO' ? null : activeMode, previewA, previewB);
                  }}
                  style={{
                    background: 'rgba(30, 41, 59, 0.6)',
                    border: '1px solid rgba(51, 65, 85, 0.6)',
                    borderRadius: '16px',
                    padding: '4px 10px',
                    color: '#cbd5e1',
                    fontSize: '11px',
                    cursor: 'pointer',
                  }}
                >
                  {prompt}
                </button>
              ))}
            </div>
          </div>

          {/* B. MAIN DISPLAY: VIEWER + RESULTS PANEL */}
          <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '16px', flex: 1, minHeight: '480px' }}>
            {/* LEFT: INTERACTIVE IMAGE VIEWER */}
            <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
              {/* Viewer Tab Bar */}
              <div
                style={{
                  padding: '10px 16px',
                  borderBottom: '1px solid rgba(51, 65, 85, 0.6)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  background: 'rgba(15, 23, 42, 0.7)',
                }}
              >
                <div style={{ display: 'flex', gap: '6px' }}>
                  {[
                    { id: 'original', label: 'Original Acquisition' },
                    { id: 'overlay', label: 'AI Intelligence Overlay' },
                    { id: 'split', label: 'Before / After Comparison' },
                  ].map((tab) => (
                    <button
                      key={tab.id}
                      onClick={() => setViewerTab(tab.id)}
                      style={{
                        padding: '6px 12px',
                        borderRadius: '6px',
                        background: viewerTab === tab.id ? '#1e293b' : 'transparent',
                        border: viewerTab === tab.id ? '1px solid #0ea5e9' : '1px solid transparent',
                        color: viewerTab === tab.id ? '#38bdf8' : '#94a3b8',
                        fontSize: '12px',
                        fontWeight: '500',
                        cursor: 'pointer',
                      }}
                    >
                      {tab.label}
                    </button>
                  ))}
                </div>

                {/* Opacity Slider for Overlay */}
                {viewerTab === 'overlay' && result?.overlay_url && (
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <Sliders size={14} color="#64748b" />
                    <span style={{ fontSize: '11px', color: '#94a3b8' }}>Opacity:</span>
                    <input
                      type="range"
                      min="0.1"
                      max="1"
                      step="0.05"
                      value={overlayOpacity}
                      onChange={(e) => setOverlayOpacity(parseFloat(e.target.value))}
                      style={{ width: '80px', accentColor: '#0ea5e9' }}
                    />
                  </div>
                )}
              </div>

              {/* Viewer Image Stage */}
              <div
                style={{
                  flex: 1,
                  background: '#070a11',
                  position: 'relative',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  overflow: 'hidden',
                  minHeight: '380px',
                }}
              >
                {/* HUD Mission Control Overlays */}
                <div style={{ position: 'absolute', top: 12, left: 14, zIndex: 10, fontSize: '11px', fontFamily: 'monospace', color: '#0ea5e9', background: 'rgba(15, 23, 42, 0.75)', padding: '4px 8px', borderRadius: '4px', border: '1px solid rgba(14, 165, 233, 0.3)' }}>
                  SENSOR: {activeScenario?.sensor || 'Sentinel-2 MSI / SAR'}
                </div>
                <div style={{ position: 'absolute', top: 12, right: 14, zIndex: 10, fontSize: '11px', fontFamily: 'monospace', color: '#34d399', background: 'rgba(15, 23, 42, 0.75)', padding: '4px 8px', borderRadius: '4px', border: '1px solid rgba(52, 211, 153, 0.3)' }}>
                  ZOOM: {Math.round(zoomLevel * 100)}% • 512x512 RES
                </div>

                {/* Loading state indicator */}
                {loading && (
                  <div style={{ position: 'absolute', inset: 0, background: 'rgba(10, 14, 23, 0.8)', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', zIndex: 20 }}>
                    <RefreshCw size={36} color="#0ea5e9" className="animate-spin" />
                    <p style={{ marginTop: '14px', fontSize: '13px', color: '#38bdf8', fontWeight: '500' }}>
                      Executing Remote Sensing Specialist Pipeline...
                    </p>
                  </div>
                )}

                {/* Display logic */}
                {viewerTab === 'original' && (
                  <img
                    src={previewA}
                    alt="Original"
                    style={{
                      maxHeight: '90%',
                      maxWidth: '90%',
                      objectFit: 'contain',
                      borderRadius: '6px',
                      transform: `scale(${zoomLevel})`,
                      transition: 'transform 0.2s',
                    }}
                  />
                )}

                {viewerTab === 'overlay' && (
                  <div style={{ position: 'relative', maxHeight: '90%', maxWidth: '90%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <img
                      src={result?.overlay_url ? `${API_BASE}${result.overlay_url}` : previewA}
                      alt="Overlay"
                      style={{
                        maxHeight: '440px',
                        maxWidth: '100%',
                        borderRadius: '6px',
                        opacity: overlayOpacity,
                        boxShadow: '0 0 25px rgba(0, 0, 0, 0.6)',
                        transform: `scale(${zoomLevel})`,
                        transition: 'transform 0.2s',
                      }}
                    />
                  </div>
                )}

                {viewerTab === 'split' && (
                  <div style={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '16px' }}>
                    {result?.overlay_url && result.before_url ? (
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', width: '100%', height: '100%' }}>
                        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                          <span style={{ fontSize: '11px', color: '#94a3b8', marginBottom: '6px' }}>BASELINE / BEFORE</span>
                          <img src={`${API_BASE}${result.before_url}`} alt="Before" style={{ width: '100%', maxHeight: '360px', objectFit: 'contain', borderRadius: '6px' }} />
                        </div>
                        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                          <span style={{ fontSize: '11px', color: '#38bdf8', marginBottom: '6px' }}>ANALYSIS OVERLAY</span>
                          <img src={`${API_BASE}${result.overlay_url}`} alt="Overlay" style={{ width: '100%', maxHeight: '360px', objectFit: 'contain', borderRadius: '6px' }} />
                        </div>
                      </div>
                    ) : (
                      <div style={{ color: '#64748b', fontSize: '13px' }}>
                        Run analysis to view side-by-side temporal and multi-sensor comparisons.
                      </div>
                    )}
                  </div>
                )}

                {/* Bottom Zoom & Reset Controls */}
                <div style={{ position: 'absolute', bottom: 12, right: 14, zIndex: 10, display: 'flex', gap: '6px' }}>
                  <button
                    onClick={() => setZoomLevel((z) => Math.min(z + 0.25, 2.5))}
                    style={{ background: 'rgba(15, 23, 42, 0.8)', border: '1px solid #334155', color: '#f8fafc', padding: '4px 8px', borderRadius: '4px', cursor: 'pointer', fontSize: '12px' }}
                  >
                    +
                  </button>
                  <button
                    onClick={() => setZoomLevel((z) => Math.max(z - 0.25, 0.75))}
                    style={{ background: 'rgba(15, 23, 42, 0.8)', border: '1px solid #334155', color: '#f8fafc', padding: '4px 8px', borderRadius: '4px', cursor: 'pointer', fontSize: '12px' }}
                  >
                    -
                  </button>
                  <button
                    onClick={() => setZoomLevel(1)}
                    style={{ background: 'rgba(15, 23, 42, 0.8)', border: '1px solid #334155', color: '#f8fafc', padding: '4px 8px', borderRadius: '4px', cursor: 'pointer', fontSize: '11px' }}
                  >
                    Reset
                  </button>
                </div>
              </div>
            </div>

            {/* RIGHT: RESULTS, STATISTICS & EXPLAINABILITY TRACE */}
            <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', overflowY: 'auto', padding: '16px' }}>
              {error && (
                <div style={{ padding: '12px', borderRadius: '8px', background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.3)', color: '#fca5a5', fontSize: '13px', marginBottom: '14px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <ShieldAlert size={18} color="#ef4444" />
                  <span>{error}</span>
                </div>
              )}

              {result ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                  {/* Task & Confidence Badge Row */}
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', paddingBottom: '12px', borderBottom: '1px solid rgba(51, 65, 85, 0.6)' }}>
                    <div>
                      <span style={{ fontSize: '10px', color: '#64748b', textTransform: 'uppercase', fontWeight: '700' }}>DETECTED TASK</span>
                      <div style={{ fontSize: '16px', fontWeight: '700', color: '#38bdf8', letterSpacing: '0.02em' }}>
                        {result.task}
                      </div>
                    </div>

                    <div style={{ textAlign: 'right' }}>
                      <span style={{ fontSize: '10px', color: '#64748b', textTransform: 'uppercase', fontWeight: '700' }}>AI CONFIDENCE</span>
                      <div style={{ fontSize: '16px', fontWeight: '700', color: '#34d399' }}>
                        {Math.round(result.confidence * 100)}%
                      </div>
                    </div>
                  </div>

                  {/* Natural Language Answer Box */}
                  <div style={{ background: 'rgba(15, 23, 42, 0.8)', border: '1px solid rgba(14, 165, 233, 0.3)', borderRadius: '8px', padding: '14px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '8px' }}>
                      <Sparkles size={16} color="#38bdf8" />
                      <span style={{ fontSize: '12px', fontWeight: '600', color: '#38bdf8' }}>SatQuery Analysis Response</span>
                    </div>
                    <p style={{ fontSize: '14px', lineHeight: '1.5', color: '#f8fafc', fontWeight: '400' }}>
                      {result.answer}
                    </p>
                  </div>

                  {/* Quantitative Statistics Grid */}
                  {result.statistics && Object.keys(result.statistics).length > 0 && (
                    <div>
                      <span style={{ fontSize: '11px', fontWeight: '700', color: '#64748b', textTransform: 'uppercase', marginBottom: '8px', display: 'block' }}>
                        Quantitative Land Cover Metrics
                      </span>
                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(110px, 1fr))', gap: '8px' }}>
                        {result.statistics.vegetation_percentage !== undefined && (
                          <div style={{ background: 'rgba(16, 185, 129, 0.1)', border: '1px solid rgba(16, 185, 129, 0.25)', borderRadius: '6px', padding: '8px' }}>
                            <div style={{ fontSize: '10px', color: '#34d399' }}>Vegetation</div>
                            <div style={{ fontSize: '15px', fontWeight: '700', color: '#f8fafc' }}>{result.statistics.vegetation_percentage}%</div>
                          </div>
                        )}
                        {result.statistics.water_percentage !== undefined && (
                          <div style={{ background: 'rgba(14, 165, 233, 0.1)', border: '1px solid rgba(14, 165, 233, 0.25)', borderRadius: '6px', padding: '8px' }}>
                            <div style={{ fontSize: '10px', color: '#38bdf8' }}>Water Bodies</div>
                            <div style={{ fontSize: '15px', fontWeight: '700', color: '#f8fafc' }}>{result.statistics.water_percentage}%</div>
                          </div>
                        )}
                        {result.statistics.built_up_percentage !== undefined && (
                          <div style={{ background: 'rgba(245, 158, 11, 0.1)', border: '1px solid rgba(245, 158, 11, 0.25)', borderRadius: '6px', padding: '8px' }}>
                            <div style={{ fontSize: '10px', color: '#fbbf24' }}>Built-up</div>
                            <div style={{ fontSize: '15px', fontWeight: '700', color: '#f8fafc' }}>{result.statistics.built_up_percentage}%</div>
                          </div>
                        )}
                        {result.statistics.total_changed_percentage !== undefined && (
                          <div style={{ background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.25)', borderRadius: '6px', padding: '8px' }}>
                            <div style={{ fontSize: '10px', color: '#f87171' }}>Changed Area</div>
                            <div style={{ fontSize: '15px', fontWeight: '700', color: '#f8fafc' }}>{result.statistics.total_changed_percentage}%</div>
                          </div>
                        )}
                        {result.statistics.fused_built_up_percentage !== undefined && (
                          <div style={{ background: 'rgba(6, 182, 212, 0.1)', border: '1px solid rgba(6, 182, 212, 0.25)', borderRadius: '6px', padding: '8px' }}>
                            <div style={{ fontSize: '10px', color: '#22d3ee' }}>Fused Built-up</div>
                            <div style={{ fontSize: '15px', fontWeight: '700', color: '#f8fafc' }}>{result.statistics.fused_built_up_percentage}%</div>
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Supporting Evidence Items */}
                  {result.evidence && result.evidence.length > 0 && (
                    <div>
                      <span style={{ fontSize: '11px', fontWeight: '700', color: '#64748b', textTransform: 'uppercase', marginBottom: '8px', display: 'block' }}>
                        Explainable Evidence Points ({result.evidence.length})
                      </span>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                        {result.evidence.map((ev, i) => (
                          <div
                            key={i}
                            style={{
                              background: 'rgba(22, 32, 50, 0.6)',
                              border: '1px solid rgba(51, 65, 85, 0.5)',
                              borderRadius: '6px',
                              padding: '8px 12px',
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'space-between',
                            }}
                          >
                            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                              <CheckCircle2 size={14} color="#10b981" />
                              <span style={{ fontSize: '12px', color: '#e2e8f0' }}>{ev.description}</span>
                            </div>
                            {ev.quadrant && (
                              <span style={{ fontSize: '10px', background: '#1e293b', color: '#38bdf8', padding: '2px 6px', borderRadius: '4px' }}>
                                {ev.quadrant}
                              </span>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Execution Trace Timeline */}
                  {result.trace && result.trace.length > 0 && (
                    <div>
                      <span style={{ fontSize: '11px', fontWeight: '700', color: '#64748b', textTransform: 'uppercase', marginBottom: '8px', display: 'block' }}>
                        Execution Trace Audit Log
                      </span>
                      <div style={{ background: '#0a0e17', borderRadius: '6px', border: '1px solid #1e293b', padding: '10px 14px' }}>
                        {result.trace.map((step, idx) => (
                          <div key={idx} style={{ display: 'flex', alignItems: 'flex-start', gap: '8px', marginBottom: idx === result.trace.length - 1 ? 0 : '6px' }}>
                            <span style={{ color: '#10b981', fontSize: '11px' }}>✓</span>
                            <span style={{ fontFamily: 'monospace', fontSize: '11px', color: '#94a3b8' }}>
                              {step}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', color: '#64748b', textAlign: 'center', padding: '20px' }}>
                  <Compass size={40} color="#334155" style={{ marginBottom: '12px' }} />
                  <p style={{ fontSize: '14px', color: '#94a3b8', fontWeight: '500' }}>Ready for Satellite Query</p>
                  <p style={{ fontSize: '12px', color: '#64748b', marginTop: '4px', maxWidth: '320px' }}>
                    Select a preset demo scenario from the left or enter a custom query to begin analysis.
                  </p>
                </div>
              )}
            </div>
          </div>
        </main>
      </div>

      {/* 3. SIH PS-26167 DOCUMENTATION MODAL */}
      {showDocsModal && (
        <div style={{ position: 'fixed', inset: 0, background: 'rgba(0, 0, 0, 0.75)', zIndex: 100, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '20px' }}>
          <div className="glass-panel" style={{ width: '640px', maxHeight: '85vh', overflowY: 'auto', padding: '24px', background: '#0f172a' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <Satellite size={20} color="#0ea5e9" />
                <h2 style={{ fontSize: '16px', fontWeight: '700', color: '#f8fafc' }}>
                  SIH 2026 / ISRO-SAC Technical Specifications
                </h2>
              </div>
              <button
                onClick={() => setShowDocsModal(false)}
                style={{ background: 'transparent', border: 'none', color: '#94a3b8', fontSize: '18px', cursor: 'pointer' }}
              >
                ✕
              </button>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '14px', fontSize: '13px', color: '#cbd5e1', lineHeight: '1.6' }}>
              <div>
                <strong style={{ color: '#38bdf8' }}>Problem Statement ID:</strong> 26167 (ISRO Space Applications Centre)
              </div>
              <div>
                <strong style={{ color: '#34d399' }}>Core Objective:</strong> Build a conversational AI interface that eliminates manual satellite image toolchains by understanding user intent and auto-routing to VQA, Grounding, Change Detection, and Multi-Sensor Fusion.
              </div>
              <div>
                <strong style={{ color: '#fbbf24' }}>Integrated Benchmark Datasets:</strong>
                <ul style={{ paddingLeft: '20px', marginTop: '6px' }}>
                  <li><strong>BigEarthNet.txt:</strong> Multi-sensor Sentinel-1 SAR + Sentinel-2 multispectral pairs with 9.6M instruction annotations.</li>
                  <li><strong>VRSBench:</strong> Remote-sensing captioning, visual grounding, and VQA benchmark.</li>
                  <li><strong>RSVQA:</strong> Remote Sensing Visual Question Answering (single-image land-cover reasoning).</li>
                  <li><strong>CDVQA:</strong> Bi-temporal Change Detection Visual Question Answering (image t1 vs t2).</li>
                </ul>
              </div>
              <div>
                <strong style={{ color: '#a855f7' }}>ISRO Multi-Sensor Adaptation:</strong>
                <p style={{ marginTop: '4px' }}>
                  Calibrated to accept Cartosat-2S (high-resolution optical) and RISAT (C-band SAR radar) co-registered imagery, fusing optical spectral response with radar double-bounce backscatter.
                </p>
              </div>
            </div>

            <div style={{ marginTop: '20px', textAlign: 'right' }}>
              <button
                onClick={() => setShowDocsModal(false)}
                style={{ background: '#0284c7', border: 'none', borderRadius: '6px', padding: '8px 16px', color: '#fff', fontSize: '13px', cursor: 'pointer' }}
              >
                Close Specifications
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
