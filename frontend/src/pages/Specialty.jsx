import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useApp } from '../context/AppContext';

const API_BASE_URL = 'http://localhost:8000/api';

export default function Specialty() {
  const navigate = useNavigate();
  const { patientId, languageLabel, setSpecialty: setAppSpecialty, t } = useApp();
  const [loading, setLoading] = useState(true);
  const [specialty, setSpecialty] = useState(null);

  useEffect(() => {
    matchSpecialty();
  }, []);

  const matchSpecialty = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/specialty-match?patient_id=${patientId}&language=${encodeURIComponent(languageLabel)}`);
      const data = await response.json();
      setSpecialty(data);
      setAppSpecialty(data);
    } catch (error) {
      console.error('Specialty matching failed:', error);
      setSpecialty({ specialty: 'General Medicine', reason: 'Default recommendation', confidence: 'Medium' });
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="page-container">
        <div className="page-content text-center">
          <div className="spinner spinner-lg" style={{ margin: '0 auto var(--space-6)' }} />
          <h3 className="heading-3 mb-2">{t('specialty_matching')}</h3>
          <p className="body-text">{t('specialty_analyzing')}</p>
        </div>
      </div>
    );
  }

  const specialtyIcons = {
    'Dermatology': '🩺', 'Ophthalmology': '👁️', 'ENT': '👂', 'Dentistry': '🦷',
    'Orthopedics': '🦴', 'Pediatrics': '👶', 'Obstetrics & Gynecology': '🤰',
    'Gastroenterology': '🫁', 'Cardiology': '❤️', 'Pulmonology': '🫁',
    'Neurology': '🧠', 'Psychiatry': '🧠', 'General Medicine': '🏥',
    'Surgery': '🔪', 'Urology': '🩺',
  };

  return (
    <div className="page-container animate-fade-in">
      <div className="page-content text-center">

        <h2 className="heading-2 mb-2">{t('specialty_title')}</h2>
        <p className="subtitle mb-8">
          {t('specialty_desc')}
        </p>

        <div className="card card-elevated" style={{
          padding: 'var(--space-8)', marginBottom: 'var(--space-6)', textAlign: 'center'
        }}>
          <div style={{ fontSize: '3rem', marginBottom: 'var(--space-4)' }}>
            {specialtyIcons[specialty?.specialty] || '🏥'}
          </div>
          <h3 className="heading-2" style={{ color: 'var(--color-primary)', marginBottom: 'var(--space-3)' }}>
            {specialty?.specialty || 'General Medicine'}
          </h3>
          {specialty?.reason && (
            <p className="body-text">{specialty.reason}</p>
          )}
          {specialty?.confidence && (
            <span className="badge badge-info mt-4" style={{ display: 'inline-flex' }}>
              {t('confidence')}: {specialty.confidence}
            </span>
          )}
        </div>

        <div className="alert alert-warning mb-6">
          ℹ️ {t('specialty_disclaimer')}
        </div>

        <button className="btn btn-primary btn-lg btn-full" onClick={() => navigate('/providers')}>
          {t('find_providers')}
        </button>

      </div>
    </div>
  );
}
