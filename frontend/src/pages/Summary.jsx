import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useApp } from '../context/AppContext';

const API_BASE_URL = 'http://localhost:8000/api';

export default function Summary() {
  const navigate = useNavigate();
  const { patientId, patient, specialty, selectedProvider, t } = useApp();
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchSummary();
  }, []);

  const fetchSummary = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/patient-summary?patient_id=${patientId}`);
      const data = await response.json();
      setSummary(data);
    } catch (error) {
      console.error('Failed to fetch summary:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="page-container">
        <div className="page-content text-center">
          <div className="spinner spinner-lg" style={{ margin: '0 auto var(--space-6)' }} />
          <h3 className="heading-3 mb-2">{t('Generating Clinical Summary...')}</h3>
          <p className="body-text">{t('Preparing your doctor-ready report')}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="page-container animate-fade-in" style={{ justifyContent: 'flex-start', paddingTop: 'var(--space-6)' }}>
      <div className="page-content-wide">

        {/* Draft Banner */}
        <div className="alert alert-warning mb-6" style={{ fontWeight: 600 }}>
          ⚠️ AI-GENERATED DRAFT — VERIFY BEFORE CLINICAL USE
        </div>

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-6)', flexWrap: 'wrap', gap: 'var(--space-4)' }}>
          <div>
            <h2 className="heading-2">{t('summary_title')}</h2>
            <p className="caption mt-2">
              {patient?.name || 'Patient'} • {patient?.age ? `Age ${patient.age}` : ''} •
              {patientId} • {specialty?.specialty || 'General Medicine'}
            </p>
          </div>
        </div>

        <div className="grid-2" style={{ gap: 'var(--space-4)', alignItems: 'start' }}>

          {/* Left Column */}
          <div className="flex flex-col gap-4">
            <SummaryCard title={t('chief_complaint')} icon="🎯" content={summary?.chief_complaint} />
            <SummaryCard title={t('hpi')} icon="📋" content={summary?.hpi} />
            <SummaryCard title={t('pmh')} icon="📂" content={summary?.past_medical_history} />
            <SummaryCard title={t('fam_hist')} icon="👨‍👩‍👧‍👦" content={summary?.family_history} />
          </div>

          {/* Right Column */}
          <div className="flex flex-col gap-4">
            <SummaryCard title={t('pers_hist')} icon="🏃" content={summary?.personal_history} />
            <SummaryCard title={t('allergies')} icon="⚠️" content={summary?.allergies} />
            <SummaryCard title={t('ros')} icon="🔍" content={summary?.review_of_systems} />

            <div className="card" style={{ padding: 'var(--space-5)' }}>
              <h4 className="heading-4 mb-2">{t('quick_info')}</h4>
              <div className="flex flex-col gap-2">
                <div className="flex items-center gap-2">
                  <span className="caption" style={{ width: 100 }}>{t('severity')}:</span>
                  <span className={`badge ${summary?.severity === 'High' ? 'badge-danger' : summary?.severity === 'Medium' ? 'badge-warning' : 'badge-success'}`}>
                    {summary?.severity || 'Unknown'}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="caption" style={{ width: 100 }}>{t('duration')}:</span>
                  <span className="body-text">{summary?.duration || 'Unknown'}</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="caption" style={{ width: 100 }}>{t('emergency')}:</span>
                  <span className={`badge ${summary?.is_emergency ? 'badge-danger' : 'badge-success'}`}>
                    {summary?.is_emergency ? 'Yes' : 'No'}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="caption" style={{ width: 100 }}>Specialty:</span>
                  <span className="body-text">{specialty?.specialty || 'General Medicine'}</span>
                </div>
              </div>
            </div>

            {selectedProvider && (
              <div className="card" style={{ padding: 'var(--space-5)', background: 'var(--color-primary-50)' }}>
                <h4 className="heading-4 mb-2">👨‍⚕️ Selected Provider</h4>
                <p className="body-text">{selectedProvider.name}</p>
                <p className="caption">{selectedProvider.facility} • {selectedProvider.distance}</p>
              </div>
            )}
          </div>
        </div>

        {/* AYUSH Section */}
        {(summary?.prakriti && summary.prakriti !== 'Not assessed') && (
          <div className="card mt-4" style={{ padding: 'var(--space-5)', borderLeft: '4px solid var(--color-success)' }}>
            <h4 className="heading-4 mb-3">🌿 AYUSH Assessment</h4>
            <div className="grid-3">
              <div>
                <div className="caption" style={{ fontWeight: 600 }}>Prakriti</div>
                <p className="body-text">{summary?.prakriti}</p>
              </div>
              <div>
                <div className="caption" style={{ fontWeight: 600 }}>Vikriti</div>
                <p className="body-text">{summary?.vikriti}</p>
              </div>
              <div>
                <div className="caption" style={{ fontWeight: 600 }}>Agni</div>
                <p className="body-text">{summary?.agni}</p>
              </div>
            </div>
          </div>
        )}

        {/* Actions */}
        <div className="flex gap-4 mt-6">
          <button className="btn btn-primary btn-full btn-lg" onClick={() => navigate('/patient-final')}>
            {t('submit_doctor')}
          </button>
        </div>

      </div>
    </div>
  );
}

function SummaryCard({ title, icon, content }) {
  if (!content || content === 'None reported' || content === 'Not recorded') {
    return (
      <div className="card" style={{ padding: 'var(--space-5)', opacity: 0.6 }}>
        <h4 className="heading-4 mb-2">{icon} {title}</h4>
        <p className="caption" style={{ fontStyle: 'italic' }}>Not reported</p>
      </div>
    );
  }
  return (
    <div className="card" style={{ padding: 'var(--space-5)' }}>
      <h4 className="heading-4 mb-2">{icon} {title}</h4>
      <p className="body-text" style={{ whiteSpace: 'pre-wrap' }}>{content}</p>
    </div>
  );
}
