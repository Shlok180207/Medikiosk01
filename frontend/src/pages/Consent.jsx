import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useApp } from '../context/AppContext';

export default function Consent() {
  const navigate = useNavigate();
  const { setConsentGiven, languageLabel, t } = useApp();
  const [showPrivacy, setShowPrivacy] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);

  const handleAgree = () => {
    setConsentGiven(true);
    navigate('/clinical-mode');
  };

  const handleDisagree = () => {
    navigate('/');
  };

  const playConsent = () => {
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(
        'MediKiosk will collect information about your current health concern and previous medical records to prepare your history for the healthcare professional. Your information will be shared only with your assigned healthcare provider. You can withdraw consent at any time.'
      );
      utterance.lang = 'en-IN';
      utterance.rate = 0.85;
      utterance.onstart = () => setIsPlaying(true);
      utterance.onend = () => setIsPlaying(false);
      window.speechSynthesis.speak(utterance);
    }
  };

  return (
    <div className="page-container animate-fade-in">
      <div className="page-content">

        <div className="text-center mb-8">
          <div style={{ fontSize: '3rem', marginBottom: 'var(--space-4)' }}>🛡️</div>
          <h2 className="heading-2" style={{ marginBottom: 'var(--space-2)' }}>{t('consent_title')}</h2>
          <p className="subtitle">{t('consent_subtitle')}</p>
        </div>

        <div className="card" style={{ padding: 'var(--space-6)', marginBottom: 'var(--space-6)' }}>
          <p className="body-text" style={{ fontSize: 'var(--font-size-lg)', lineHeight: 1.8, marginBottom: 'var(--space-6)' }}>
            {t('consent_info_1')}<strong>{t('consent_info_2')}</strong>{t('consent_info_3')}
            <strong>{t('consent_info_4')}</strong>{t('consent_info_5')}
          </p>

          <div className="flex flex-col gap-4">
            <div className="flex items-center gap-3">
              <span style={{
                width: 40, height: 40, borderRadius: 'var(--radius-full)',
                background: 'var(--color-info-bg)', display: 'flex', alignItems: 'center', justifyContent: 'center',
                flexShrink: 0, fontSize: '1.1rem'
              }}>📋</span>
              <div>
                <div style={{ fontWeight: 600, color: 'var(--color-text)' }}>{t('what_we_collect')}</div>
                <p className="caption">{t('what_we_collect_desc')}</p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <span style={{
                width: 40, height: 40, borderRadius: 'var(--radius-full)',
                background: 'var(--color-info-bg)', display: 'flex', alignItems: 'center', justifyContent: 'center',
                flexShrink: 0, fontSize: '1.1rem'
              }}>🎯</span>
              <div>
                <div style={{ fontWeight: 600, color: 'var(--color-text)' }}>{t('why_we_collect')}</div>
                <p className="caption">{t('why_we_collect_desc')}</p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <span style={{
                width: 40, height: 40, borderRadius: 'var(--radius-full)',
                background: 'var(--color-info-bg)', display: 'flex', alignItems: 'center', justifyContent: 'center',
                flexShrink: 0, fontSize: '1.1rem'
              }}>👨‍⚕️</span>
              <div>
                <div style={{ fontWeight: 600, color: 'var(--color-text)' }}>{t('who_can_access')}</div>
                <p className="caption">{t('who_can_access_desc')}</p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <span style={{
                width: 40, height: 40, borderRadius: 'var(--radius-full)',
                background: 'var(--color-info-bg)', display: 'flex', alignItems: 'center', justifyContent: 'center',
                flexShrink: 0, fontSize: '1.1rem'
              }}>🔒</span>
              <div>
                <div style={{ fontWeight: 600, color: 'var(--color-text)' }}>{t('how_its_used')}</div>
                <p className="caption">{t('how_its_used_desc')}</p>
              </div>
            </div>
          </div>
        </div>

        {/* Audio playback */}
        <button className="btn btn-ghost btn-full mb-4" onClick={playConsent}
          style={{ border: '1px solid var(--color-border)' }}>
          {isPlaying ? t('playing') : t('listen_consent')}
        </button>

        {/* Privacy toggle */}
        <button className="btn btn-ghost btn-full mb-6" onClick={() => setShowPrivacy(!showPrivacy)}
          style={{ fontSize: 'var(--font-size-sm)' }}>
          {showPrivacy ? '▾' : '▸'} {t('view_privacy')}
        </button>

        {showPrivacy && (
          <div className="card mb-6" style={{ padding: 'var(--space-4)', background: 'var(--color-bg)' }}>
            <p className="caption" style={{ lineHeight: 1.8 }}>
              {t('privacy_info')}
            </p>
          </div>
        )}

        {/* Action Buttons */}
        <div className="flex gap-4">
          <button className="btn btn-secondary btn-full btn-lg" onClick={handleDisagree}>
            {t('decline')}
          </button>
          <button className="btn btn-primary btn-full btn-lg" onClick={handleAgree}>
            {t('accept_continue')}
          </button>
        </div>

      </div>
    </div>
  );
}
