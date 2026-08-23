import React from 'react';
import { useNavigate } from 'react-router-dom';
import { useApp } from '../context/AppContext';

export default function LanguageSelect() {
  const navigate = useNavigate();
  const { languages, language, setLanguage } = useApp();

  const handleSelect = (lang) => {
    setLanguage(lang);
    navigate('/patient-id');
  };

  return (
    <div className="page-container animate-fade-in">
      <div className="page-content text-center">

        <h2 className="heading-2" style={{ marginBottom: 'var(--space-2)' }}>
          Choose your language
        </h2>
        <p className="subtitle" style={{ marginBottom: 'var(--space-8)' }}>
          अपनी भाषा चुनें
        </p>

        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))',
          gap: 'var(--space-4)',
          marginBottom: 'var(--space-8)'
        }}>
          {languages.map((lang) => (
            <button
              key={lang.code}
              className={`card card-interactive ${language?.code === lang.code ? 'active' : ''}`}
              onClick={() => handleSelect(lang)}
              style={{
                display: 'flex', flexDirection: 'column', alignItems: 'center',
                padding: 'var(--space-5) var(--space-4)', gap: 'var(--space-1)',
                textAlign: 'center'
              }}
            >
              <span style={{ fontSize: 'var(--font-size-2xl)', fontWeight: 700, color: 'var(--color-text)' }}>
                {lang.native}
              </span>
              <span className="caption">{lang.label}</span>
            </button>
          ))}
        </div>

        {/* Voice + Touch hints */}
        <div className="flex flex-center gap-6" style={{ flexWrap: 'wrap' }}>
          <div className="flex items-center gap-2">
            <span style={{ fontSize: '1.5rem' }}>🎙️</span>
            <span className="body-text">You can answer by speaking</span>
          </div>
          <div className="flex items-center gap-2">
            <span style={{ fontSize: '1.5rem' }}>👆</span>
            <span className="body-text">You can also tap answers</span>
          </div>
        </div>

      </div>
    </div>
  );
}
