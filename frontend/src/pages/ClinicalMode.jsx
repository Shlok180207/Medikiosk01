import React from 'react';
import { useNavigate } from 'react-router-dom';
import { useApp } from '../context/AppContext';

export default function ClinicalMode() {
  const navigate = useNavigate();
  const { setClinicalMode, t } = useApp();

  const handleSelect = (mode) => {
    setClinicalMode(mode);
    navigate('/kiosk');
  };

  return (
    <div className="page-container animate-fade-in">
      <div className="page-content text-center">

        <h2 className="heading-2" style={{ marginBottom: 'var(--space-2)' }}>
          {t('mode_title')}
        </h2>
        <p className="subtitle" style={{ marginBottom: 'var(--space-8)' }}>
          {t('mode_subtitle')}
        </p>

        <div className="flex flex-col gap-4">
          <button
            className="card card-interactive"
            onClick={() => handleSelect('allopathic')}
            style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-5)', padding: 'var(--space-8)', textAlign: 'left' }}
          >
            <div style={{
              width: 72, height: 72, borderRadius: 'var(--radius-xl)',
              background: 'linear-gradient(135deg, #0891b2, #0284c7)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: '2rem', flexShrink: 0, color: 'white'
            }}>🏥</div>
            <div>
              <div className="heading-3">{t('allo_opd')}</div>
              <p className="body-text mt-2">{t('allo_opd_desc')}</p>
            </div>
          </button>

          <button
            className="card card-interactive"
            onClick={() => handleSelect('ayush')}
            style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-5)', padding: 'var(--space-8)', textAlign: 'left' }}
          >
            <div style={{
              width: 72, height: 72, borderRadius: 'var(--radius-xl)',
              background: 'linear-gradient(135deg, #059669, #047857)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: '2rem', flexShrink: 0, color: 'white'
            }}>🌿</div>
            <div>
              <div className="heading-3">{t('ayush_opd')}</div>
              <p className="body-text mt-2">
                {t('ayush_opd_desc')}
              </p>
            </div>
          </button>
        </div>

      </div>
    </div>
  );
}
