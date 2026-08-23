import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useApp } from '../context/AppContext';

export default function PatientID() {
  const navigate = useNavigate();
  const { setPatient, loadDemoPatient, t } = useApp();
  const [mode, setMode] = useState(null); // 'scan', 'enter', 'new'
  const [abhaInput, setAbhaInput] = useState('');
  const [newPatient, setNewPatient] = useState({ name: '', age: '', gender: 'Male', phone: '' });

  const handleDemo = () => {
    loadDemoPatient();
    navigate('/consent');
  };

  const handleSubmitABHA = () => {
    if (!abhaInput.trim()) return;
    setPatient({ name: 'ABHA Patient', abhaId: abhaInput.trim(), age: '', gender: '', phone: '' });
    navigate('/consent');
  };

  const handleSubmitNew = () => {
    if (!newPatient.name.trim()) return;
    setPatient({ ...newPatient, abhaId: null });
    navigate('/consent');
  };

  return (
    <div className="page-container animate-fade-in">
      <div className="page-content">
        <div className="text-center" style={{ marginBottom: 'var(--space-8)' }}>
          <h2 className="heading-2" style={{ marginBottom: 'var(--space-2)' }}>{t('patient_id_subtitle')}</h2>
          <p className="subtitle">{t('patient_id_title')}</p>
        </div>

        {!mode && (
          <>
            <div className="flex flex-col gap-4" style={{ marginBottom: 'var(--space-6)' }}>
              <button className="card card-interactive" onClick={() => setMode('scan')}
                style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-4)', padding: 'var(--space-6)' }}>
                <div style={{
                  width: 56, height: 56, borderRadius: 'var(--radius-lg)',
                  background: 'var(--color-primary-50)', display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontSize: '1.5rem', flexShrink: 0
                }}>📱</div>
                <div style={{ textAlign: 'left' }}>
                  <div className="heading-4">{t('scan_abha')}</div>
                  <p className="caption">{t('scan_abha_desc')}</p>
                </div>
              </button>

              <button className="card card-interactive" onClick={() => setMode('enter')}
                style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-4)', padding: 'var(--space-6)' }}>
                <div style={{
                  width: 56, height: 56, borderRadius: 'var(--radius-lg)',
                  background: 'var(--color-primary-50)', display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontSize: '1.5rem', flexShrink: 0
                }}>🔢</div>
                <div style={{ textAlign: 'left' }}>
                  <div className="heading-4">{t('enter_abha')}</div>
                  <p className="caption">{t('enter_abha_desc')}</p>
                </div>
              </button>

              <button className="card card-interactive" onClick={() => setMode('new')}
                style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-4)', padding: 'var(--space-6)' }}>
                <div style={{
                  width: 56, height: 56, borderRadius: 'var(--radius-lg)',
                  background: 'var(--color-primary-50)', display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontSize: '1.5rem', flexShrink: 0
                }}>👤</div>
                <div style={{ textAlign: 'left' }}>
                  <div className="heading-4">{t('continue_no_abha')}</div>
                  <p className="caption">{t('continue_no_abha_desc')}</p>
                </div>
              </button>
            </div>

            <div className="divider" />

            <button className="btn btn-outline btn-full" onClick={handleDemo}
              style={{ borderColor: 'var(--color-success)', color: 'var(--color-success)' }}>
              🧪 {t('demo_patient')}
            </button>
            <p className="caption text-center mt-2">{t('demo_patient_desc')}</p>
          </>
        )}

        {/* ABHA Scan Mode */}
        {mode === 'scan' && (
          <div className="card text-center" style={{ padding: 'var(--space-8)' }}>
            <div style={{
              width: 200, height: 200, margin: '0 auto var(--space-6)',
              border: '3px dashed var(--color-border)', borderRadius: 'var(--radius-xl)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              background: 'var(--color-bg)', fontSize: '3rem'
            }}>
              📷
            </div>
            <p className="body-text mb-4">Position your ABHA QR code in the frame</p>
            <p className="caption mb-6">(Camera scanner — demo placeholder)</p>
            <div className="flex gap-4">
              <button className="btn btn-secondary btn-full" onClick={() => setMode(null)}>← Back</button>
              <button className="btn btn-primary btn-full" onClick={handleDemo}>{t('demo_patient')}</button>
            </div>
          </div>
        )}

        {/* Enter ABHA Mode */}
        {mode === 'enter' && (
          <div className="card" style={{ padding: 'var(--space-6)' }}>
            <h3 className="heading-4 mb-4">Enter your ABHA ID</h3>
            <input className="input mb-4" placeholder="e.g. 1234-5678-9012-3456"
              value={abhaInput} onChange={e => setAbhaInput(e.target.value)} />
            <div className="flex gap-4">
              <button className="btn btn-secondary btn-full" onClick={() => setMode(null)}>← Back</button>
              <button className="btn btn-primary btn-full" onClick={handleSubmitABHA} disabled={!abhaInput.trim()}>
                Continue →
              </button>
            </div>
          </div>
        )}

        {/* New Patient Mode */}
        {mode === 'new' && (
          <div className="card" style={{ padding: 'var(--space-6)' }}>
            <h3 className="heading-4 mb-4">New Patient Registration</h3>
            <div className="flex flex-col gap-4 mb-6">
              <input className="input" placeholder="Full Name *"
                value={newPatient.name} onChange={e => setNewPatient(p => ({ ...p, name: e.target.value }))} />
              <div className="grid-2">
                <input className="input" placeholder="Age" type="number"
                  value={newPatient.age} onChange={e => setNewPatient(p => ({ ...p, age: e.target.value }))} />
                <select className="input" value={newPatient.gender}
                  onChange={e => setNewPatient(p => ({ ...p, gender: e.target.value }))}>
                  <option value="Male">Male</option>
                  <option value="Female">Female</option>
                  <option value="Other">Other</option>
                </select>
              </div>
              <input className="input" placeholder="Phone Number"
                value={newPatient.phone} onChange={e => setNewPatient(p => ({ ...p, phone: e.target.value }))} />
            </div>
            <div className="flex gap-4">
              <button className="btn btn-secondary btn-full" onClick={() => setMode(null)}>← Back</button>
              <button className="btn btn-primary btn-full" onClick={handleSubmitNew} disabled={!newPatient.name.trim()}>
                Continue →
              </button>
            </div>
          </div>
        )}

      </div>
    </div>
  );
}
