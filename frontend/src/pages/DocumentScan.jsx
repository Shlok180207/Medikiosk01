import React, { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { useApp } from '../context/AppContext';

const API_BASE_URL = 'http://localhost:8000/api';

export default function DocumentScan() {
  const navigate = useNavigate();
  const { patientId, documents, setDocuments, languageLabel, isAyush, t } = useApp();
  const [isProcessing, setIsProcessing] = useState(false);
  const [extractedDoc, setExtractedDoc] = useState(null);
  const fileInputRef = useRef(null);

  const triggerFinalizeIntake = () => {
    if (patientId) {
      const formData = new FormData();
      formData.append('patient_id', patientId);
      formData.append('language', languageLabel || 'English');
      formData.append('is_ayush', Boolean(isAyush));
      fetch(`${API_BASE_URL}/finalize-intake`, { method: 'POST', body: formData }).catch(err => console.log('Finalize intake note:', err));
    }
  };

  const handleFileUpload = async (e) => {
    if (e.target.files.length === 0) return;
    const file = e.target.files[0];
    setIsProcessing(true);
    setExtractedDoc(null);

    try {
      const formData = new FormData();
      formData.append('file', file);
      if (patientId) formData.append('patient_id', patientId);

      const response = await fetch(`${API_BASE_URL}/process-document`, { method: 'POST', body: formData });
      const data = await response.json();

      if (data.extracted_document) {
        setExtractedDoc(data.extracted_document);
        setDocuments(prev => [...prev, { filename: file.name, ...data.extracted_document }]);
      }
    } catch (error) {
      console.error('Document processing failed:', error);
      setExtractedDoc({ error: true, summary: 'Failed to process document. Please try again.' });
    } finally {
      setIsProcessing(false);
      e.target.value = '';
    }
  };

  const handleSkip = () => {
    triggerFinalizeIntake();
    navigate('/specialty');
  };

  const handleContinue = () => {
    setExtractedDoc(null); // Reset for another scan
  };

  const handleDone = () => {
    triggerFinalizeIntake();
    navigate('/specialty');
  };

  return (
    <div className="page-container animate-fade-in">
      <div className="page-content">

        <div className="text-center mb-8">
          <div style={{ fontSize: '3rem', marginBottom: 'var(--space-4)' }}>📄</div>
          <h2 className="heading-2" style={{ marginBottom: 'var(--space-2)' }}>
            {t('docs_title')}
          </h2>
          <p className="subtitle">
            {t('docs_subtitle')}
          </p>
        </div>

        {/* No extraction shown — show upload options */}
        {!extractedDoc && !isProcessing && (
          <>
            <div className="flex flex-col gap-4 mb-6">
              <button className="card card-interactive" onClick={() => fileInputRef.current?.click()}
                style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-4)', padding: 'var(--space-6)' }}>
                <div style={{
                  width: 56, height: 56, borderRadius: 'var(--radius-lg)',
                  background: 'var(--color-primary-50)', display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontSize: '1.5rem', flexShrink: 0
                }}>📷</div>
                <div style={{ textAlign: 'left' }}>
                  <div className="heading-4">{t('scan_doc')}</div>
                  <p className="caption">{t('scan_doc_desc')}</p>
                </div>
              </button>
            </div>

            <input ref={fileInputRef} type="file" accept="image/*,.pdf" capture="environment"
              onChange={handleFileUpload} style={{ display: 'none' }} />

            <p className="caption text-center mb-4">
              {t('docs_supported')}
            </p>

            {documents.length > 0 && (
              <div className="card mb-4" style={{ padding: 'var(--space-4)' }}>
                <div className="heading-4 mb-2">📁 {documents.length} {t('docs_processed')}</div>
                {documents.map((doc, i) => (
                  <div key={i} className="flex items-center gap-2 mb-2">
                    <span className="badge badge-success">✓</span>
                    <span className="body-text">{doc.filename}</span>
                  </div>
                ))}
              </div>
            )}

            <div className="flex gap-4">
              <button className="btn btn-secondary btn-full" onClick={handleSkip}>
                {documents.length > 0 ? t('done_docs') : t('skip_docs')}
              </button>
            </div>
          </>
        )}

        {/* Processing */}
        {isProcessing && (
          <div className="card text-center" style={{ padding: 'var(--space-10)' }}>
            <div className="spinner spinner-lg" style={{ margin: '0 auto var(--space-6)' }} />
            <p className="heading-4 mb-2">{t('processing_doc')}</p>
            <p className="body-text">OCR → Medical NLP → Structured Data Extraction</p>
          </div>
        )}

        {/* Upload Success Result */}
        {extractedDoc && !extractedDoc.error && (
          <div className="animate-slide-up">
            <div className="alert alert-success mb-4">
              ✅ Document uploaded securely. Our AI is analyzing it in the background.
            </div>

            <div className="flex gap-3 mt-6">
              <button className="btn btn-success btn-full" onClick={handleDone}>{t('confirm_continue')}</button>
              <button className="btn btn-outline btn-full" onClick={handleContinue}>{t('scan_another')}</button>
            </div>
          </div>
        )}

        {extractedDoc?.error && (
          <div className="alert alert-danger mb-4">
            ❌ {extractedDoc.summary}
          </div>
        )}

      </div>
    </div>
  );
}
