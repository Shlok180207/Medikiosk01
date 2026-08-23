import React from 'react';
import { useNavigate } from 'react-router-dom';
import { useApp } from '../context/AppContext';

export default function PatientFinal() {
  const navigate = useNavigate();
  const { patient, patientId, specialty, selectedProvider, documents, resetFlow, t } = useApp();

  const checkmarks = [
    { label: t('checklist_history'), done: true },
    { label: `${documents.length} ${t('checklist_docs')}`, done: documents.length > 0 },
    { label: t('checklist_meds'), done: documents.length > 0 },
    { label: t('checklist_summary'), done: true },
    { label: t('checklist_provider'), done: !!selectedProvider },
  ];

  const handleNewSession = () => {
    resetFlow();
    navigate('/');
  };

  return (
    <div className="page-container animate-fade-in">
      <div className="page-content text-center">

        <div style={{
          width: 100, height: 100, borderRadius: 'var(--radius-full)',
          background: 'var(--color-success-bg)', display: 'flex', alignItems: 'center', justifyContent: 'center',
          margin: '0 auto var(--space-6)', fontSize: '3rem'
        }}>
          ✅
        </div>

        <h2 className="heading-2 mb-2">{t('final_title')}</h2>
        <p className="subtitle mb-8">
          {t('final_subtitle')}
        </p>

        {/* Checklist */}
        <div className="card" style={{ padding: 'var(--space-6)', marginBottom: 'var(--space-6)', textAlign: 'left' }}>
          {checkmarks.map((item, i) => (
            <div key={i} className="flex items-center gap-3 mb-3">
              <span style={{
                width: 28, height: 28, borderRadius: 'var(--radius-full)',
                background: item.done ? 'var(--color-success)' : 'var(--color-border)',
                color: 'white', display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: 'var(--font-size-sm)', fontWeight: 700, flexShrink: 0
              }}>
                {item.done ? '✓' : '–'}
              </span>
              <span className="body-text" style={{ opacity: item.done ? 1 : 0.5 }}>{item.label}</span>
            </div>
          ))}
        </div>

        {/* Provider info */}
        {selectedProvider && (
          <div className="card mb-6" style={{ padding: 'var(--space-5)', background: 'var(--color-primary-50)', textAlign: 'left' }}>
            <div className="heading-4 mb-2">👨‍⚕️ Your Provider</div>
            <p className="body-text">{selectedProvider.name}</p>
            <p className="caption">{selectedProvider.facility} • {selectedProvider.timing}</p>
          </div>
        )}

        {/* Actions */}
        <div className="grid-2 mb-6">
          <button className="btn btn-outline" onClick={() => navigate('/summary')}>{t('view_summary')}</button>
          <button className="btn btn-outline" onClick={() => navigate('/providers')}>{t('view_provider')}</button>
        </div>

        <button className="btn btn-primary btn-full btn-lg" onClick={handleNewSession}>
          {t('new_session')}
        </button>

        <p className="caption mt-6" style={{ maxWidth: 480, margin: 'var(--space-6) auto 0' }}>
          <strong>ABDM provides the digital health ecosystem.</strong><br />
          MediKiosk solves the first-mile clinical intake problem.
        </p>

      </div>
    </div>
  );
}
