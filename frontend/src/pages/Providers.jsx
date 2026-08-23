import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useApp } from '../context/AppContext';

const DEMO_PROVIDERS = [
  {
    id: 1, name: 'Dr. Rahul Sharma', specialty: 'General Medicine',
    distance: '1.4 km', facility: 'Government Hospital', type: 'government',
    availability: 'OPD Available', availabilityColor: '#059669',
    timing: 'Today: 10:00 AM – 1:00 PM',
    qualifications: 'MBBS, MD (Internal Medicine)',
  },
  {
    id: 2, name: 'Dr. Priya Mehta', specialty: 'General Medicine',
    distance: '2.8 km', facility: 'Clinic', type: 'clinic',
    availability: 'Appointment Required', availabilityColor: '#d97706',
    timing: 'Next: Tomorrow 9:00 AM',
    qualifications: 'MBBS, DNB',
  },
  {
    id: 3, name: 'Dr. Anand Verma', specialty: 'General Medicine',
    distance: '4.2 km', facility: 'Multi-Specialty Hospital', type: 'hospital',
    availability: 'OPD Available', availabilityColor: '#059669',
    timing: 'Today: 2:00 PM – 5:00 PM',
    qualifications: 'MBBS, MD, DM (Cardiology)',
  },
];

export default function Providers() {
  const navigate = useNavigate();
  const { specialty, setSelectedProvider, t } = useApp();
  const [selected, setSelected] = useState(null);

  const handleSelect = (provider) => {
    setSelected(provider.id);
    setSelectedProvider(provider);
  };

  const handleContinue = () => {
    navigate('/summary');
  };

  return (
    <div className="page-container animate-fade-in" style={{ justifyContent: 'flex-start', paddingTop: 'var(--space-8)' }}>
      <div className="page-content-wide">

        <div className="text-center mb-6">
          <h2 className="heading-2 mb-2">{t('providers_title')}</h2>
          <p className="subtitle">
            {t('showing_providers')} <strong>{specialty?.specialty || 'General Medicine'}</strong>
          </p>
        </div>

        <div className="alert alert-info mb-6">
          🧪 <strong>Demo Availability</strong> — Provider data shown below is simulated for demonstration purposes.
        </div>

        <div className="flex flex-col gap-4 mb-6">
          {DEMO_PROVIDERS.map(provider => (
            <div
              key={provider.id}
              className={`card card-interactive ${selected === provider.id ? 'active' : ''}`}
              onClick={() => handleSelect(provider)}
              style={{ padding: 'var(--space-6)' }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 'var(--space-4)' }}>
                <div style={{ flex: 1 }}>
                  <h3 className="heading-4" style={{ marginBottom: 'var(--space-1)' }}>{provider.name}</h3>
                  <p className="body-text" style={{ marginBottom: 'var(--space-2)' }}>{provider.specialty}</p>
                  <p className="caption" style={{ marginBottom: 'var(--space-1)' }}>{provider.qualifications}</p>
                  <div className="flex items-center gap-4 mt-2" style={{ flexWrap: 'wrap' }}>
                    <span className="caption">📍 {provider.distance}</span>
                    <span className="caption">🏥 {provider.facility}</span>
                  </div>
                </div>
                <div style={{ textAlign: 'right' }}>
                  <div style={{
                    display: 'inline-flex', alignItems: 'center', gap: 'var(--space-2)',
                    padding: 'var(--space-1) var(--space-3)', borderRadius: 'var(--radius-full)',
                    background: provider.availabilityColor === '#059669' ? 'var(--color-success-bg)' : 'var(--color-warning-bg)',
                    color: provider.availabilityColor, fontSize: 'var(--font-size-sm)', fontWeight: 600
                  }}>
                    <span style={{ width: 8, height: 8, borderRadius: '50%', background: provider.availabilityColor }} />
                    {provider.availability}
                  </div>
                  <p className="caption mt-2">{provider.timing}</p>
                </div>
              </div>

              {selected === provider.id && (
                <div className="flex gap-3 mt-4" style={{ borderTop: '1px solid var(--color-border)', paddingTop: 'var(--space-4)' }}>
                  <button className="btn btn-primary btn-sm" style={{ flex: 1 }} onClick={handleContinue}>
                    {t('select_continue')}
                  </button>
                  <button className="btn btn-secondary btn-sm" style={{ flex: 1 }}>
                    {t('directions')}
                  </button>
                </div>
              )}
            </div>
          ))}
        </div>

        {!selected && (
          <button className="btn btn-ghost btn-full" onClick={handleContinue}>
            {t('skip_providers')}
          </button>
        )}

      </div>
    </div>
  );
}
