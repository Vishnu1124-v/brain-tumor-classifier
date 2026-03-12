import React, { useState, useCallback } from 'react';
import axios from 'axios';
import './App.css';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

const CLASS_COLORS = {
  glioma:     { bg: '#fef2f2', border: '#ef4444', text: '#b91c1c', badge: '#ef4444' },
  meningioma: { bg: '#fff7ed', border: '#f97316', text: '#c2410c', badge: '#f97316' },
  notumor:    { bg: '#f0fdf4', border: '#22c55e', text: '#15803d', badge: '#22c55e' },
  pituitary:  { bg: '#eff6ff', border: '#3b82f6', text: '#1d4ed8', badge: '#3b82f6' },
};

const SEVERITY_ICONS = {
  High:   '🔴',
  Medium: '🟠',
  None:   '🟢',
};

export default function App() {
  const [image,      setImage]      = useState(null);
  const [preview,    setPreview]    = useState(null);
  const [result,     setResult]     = useState(null);
  const [loading,    setLoading]    = useState(false);
  const [error,      setError]      = useState(null);
  const [dragOver,   setDragOver]   = useState(false);

  const handleFile = useCallback((file) => {
    if (!file || !file.type.startsWith('image/')) {
      setError('Please upload a valid image file.');
      return;
    }
    setImage(file);
    setPreview(URL.createObjectURL(file));
    setResult(null);
    setError(null);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
    handleFile(e.dataTransfer.files[0]);
  }, [handleFile]);

  const handlePredict = async () => {
    if (!image) return;
    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('image', image);

    try {
      const res = await axios.post(`${API_URL}/predict`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setResult(res.data);
    } catch (err) {
      setError('Prediction failed. Make sure Flask backend is running on port 5000.');
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setImage(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  const colors = result ? (CLASS_COLORS[result.prediction] || CLASS_COLORS.notumor) : null;

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-inner">
          <div className="logo">
            <span className="logo-icon">🧠</span>
            <div>
              <h1 className="logo-title">NeuraScan</h1>
              <p className="logo-sub">Brain Tumor Classification AI</p>
            </div>
          </div>
          <div className="header-badges">
            <span className="badge badge-green">97.14% Accuracy</span>
            <span className="badge badge-blue">EfficientNetB0</span>
            <span className="badge badge-purple">CNN + SVM Hybrid</span>
          </div>
        </div>
      </header>

      <main className="main">
        {/* Upload Section */}
        <section className="card upload-card">
          <h2 className="section-title">Upload MRI Scan</h2>
          <p className="section-sub">Upload a brain MRI image to classify tumor type</p>

          <div
            className={`dropzone ${dragOver ? 'dragover' : ''} ${preview ? 'has-image' : ''}`}
            onDrop={handleDrop}
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onClick={() => !preview && document.getElementById('fileInput').click()}
          >
            {preview ? (
              <div className="preview-container">
                <img src={preview} alt="MRI Preview" className="preview-img" />
                <div className="preview-overlay">
                  <button className="btn-change" onClick={(e) => { e.stopPropagation(); document.getElementById('fileInput').click(); }}>
                    Change Image
                  </button>
                </div>
              </div>
            ) : (
              <div className="dropzone-content">
                <div className="upload-icon">📂</div>
                <p className="upload-text">Drop MRI image here</p>
                <p className="upload-sub">or click to browse</p>
                <p className="upload-formats">Supports: JPG, PNG, JPEG</p>
              </div>
            )}
          </div>

          <input
            id="fileInput"
            type="file"
            accept="image/*"
            style={{ display: 'none' }}
            onChange={(e) => handleFile(e.target.files[0])}
          />

          {error && <div className="error-box">⚠️ {error}</div>}

          <div className="btn-row">
            {preview && (
              <>
                <button className="btn btn-primary" onClick={handlePredict} disabled={loading}>
                  {loading ? <><span className="spinner"></span> Analyzing...</> : '🔬 Analyze MRI'}
                </button>
                <button className="btn btn-secondary" onClick={reset}>🔄 Reset</button>
              </>
            )}
          </div>
        </section>

        {/* Results Section */}
        {result && (
          <section className="results-grid">

            {/* Main Prediction */}
            <div className="card result-card" style={{ borderColor: colors.border, background: colors.bg }}>
              <div className="result-header">
                <span className="result-icon">🎯</span>
                <h3>Diagnosis Result</h3>
              </div>
              <div className="result-class" style={{ color: colors.text }}>
                {result.tumor_info?.full_name || result.prediction}
              </div>
              <div className="confidence-bar-wrap">
                <div className="confidence-label">
                  <span>Confidence</span>
                  <span style={{ color: colors.badge }}>{result.confidence}%</span>
                </div>
                <div className="confidence-bar">
                  <div
                    className="confidence-fill"
                    style={{ width: `${result.confidence}%`, background: colors.badge }}
                  ></div>
                </div>
              </div>
              <div className="severity-row">
                <span>{SEVERITY_ICONS[result.tumor_info?.severity]} Severity: <strong>{result.tumor_info?.severity}</strong></span>
              </div>
              <div className="urgency-box" style={{ borderColor: colors.border }}>
                {result.tumor_info?.urgency}
              </div>
            </div>

            {/* Tumor Info */}
            <div className="card info-card">
              <div className="result-header">
                <span className="result-icon">📋</span>
                <h3>Medical Information</h3>
              </div>
              <div className="info-row">
                <span className="info-label">Description</span>
                <p className="info-value">{result.tumor_info?.description}</p>
              </div>
              <div className="info-row">
                <span className="info-label">Recommended Treatment</span>
                <p className="info-value">{result.tumor_info?.treatment}</p>
              </div>
              <div className="info-row">
                <span className="info-label">DL Model</span>
                <p className="info-value">{result.dl_prediction} ({result.dl_confidence}%)</p>
              </div>
              <div className="info-row">
                <span className="info-label">SVM Model</span>
                <p className="info-value">{result.svm_prediction} ({result.svm_confidence}%)</p>
              </div>
            </div>

            {/* All Class Probabilities */}
            <div className="card prob-card">
              <div className="result-header">
                <span className="result-icon">📊</span>
                <h3>Class Probabilities</h3>
              </div>
              {Object.entries(result.all_probabilities)
                .sort((a, b) => b[1] - a[1])
                .map(([cls, prob]) => (
                  <div key={cls} className="prob-row">
                    <span className="prob-label">{cls.charAt(0).toUpperCase() + cls.slice(1)}</span>
                    <div className="prob-bar">
                      <div
                        className="prob-fill"
                        style={{
                          width: `${prob}%`,
                          background: CLASS_COLORS[cls]?.badge || '#6366f1'
                        }}
                      ></div>
                    </div>
                    <span className="prob-value">{prob}%</span>
                  </div>
                ))}
            </div>

            {/* Grad-CAM */}
            {result.gradcam && (
              <div className="card gradcam-card">
                <div className="result-header">
                  <span className="result-icon">🔥</span>
                  <h3>Grad-CAM Visualization</h3>
                </div>
                <p className="gradcam-sub">Highlighted regions show where the AI focused</p>
                <div className="gradcam-images">
                  <div className="gradcam-item">
                    <img src={preview} alt="Original" className="gradcam-img" />
                    <span className="gradcam-label">Original MRI</span>
                  </div>
                  <div className="gradcam-item">
                    <img src={`data:image/png;base64,${result.gradcam}`} alt="Grad-CAM" className="gradcam-img" />
                    <span className="gradcam-label">AI Focus Map</span>
                  </div>
                </div>
              </div>
            )}

          </section>
        )}

        {/* Disclaimer */}
        <div className="disclaimer">
          ⚕️ <strong>Medical Disclaimer:</strong> This tool is for research and educational purposes only.
          Always consult a qualified medical professional for diagnosis and treatment.
        </div>
      </main>
    </div>
  );
}
