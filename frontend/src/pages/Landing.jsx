import React from 'react';
import { useNavigate } from 'react-router-dom';

export default function Landing() {
  const navigate = useNavigate();

  return (
    <div className="page-container" style={{ background: 'linear-gradient(180deg, #ecfeff 0%, #f8fafc 50%)' }}>
      <div className="page-content text-center animate-fade-in">

        {/* Logo / Brand */}
        <div style={{ marginBottom: 'var(--space-6)' }}>
          <div style={{
            display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
            width: 80, height: 80, borderRadius: 'var(--radius-2xl)',
            background: 'linear-gradient(135deg, #0891b2, #0284c7)',
            boxShadow: '0 8px 30px rgba(8, 145, 178, 0.3)', marginBottom: 'var(--space-4)'
          }}>
            <span style={{ fontSize: '2.5rem' }}>🏥</span>
          </div>
        </div>

        <h1 className="heading-1" style={{ marginBottom: 'var(--space-3)' }}>
          MediKiosk
        </h1>
        <p className="subtitle" style={{ maxWidth: 480, margin: '0 auto var(--space-10)' }}>
          Your health history, organized before you meet your doctor.
        </p>

        {/* Benefits */}
        <div className="grid-3" style={{ marginBottom: 'var(--space-10)', textAlign: 'center' }}>
          <div className="card" style={{ padding: 'var(--space-6)' }}>
            <div style={{ fontSize: '2rem', marginBottom: 'var(--space-3)' }}>🧠</div>
            <div className="heading-4" style={{ marginBottom: 'var(--space-2)', fontSize: 'var(--font-size-base)' }}>
              AI-Powered Clinical History
            </div>
            <p className="caption">Voice-enabled smart intake in your language</p>
          </div>
          <div className="card" style={{ padding: 'var(--space-6)' }}>
            <div style={{ fontSize: '2rem', marginBottom: 'var(--space-3)' }}>📄</div>
            <div className="heading-4" style={{ marginBottom: 'var(--space-2)', fontSize: 'var(--font-size-base)' }}>
              Document Digitization
            </div>
            <p className="caption">Scan prescriptions & lab reports instantly</p>
          </div>
          <div className="card" style={{ padding: 'var(--space-6)' }}>
            <div style={{ fontSize: '2rem', marginBottom: 'var(--space-3)' }}>🧭</div>
            <div className="heading-4" style={{ marginBottom: 'var(--space-2)', fontSize: 'var(--font-size-base)' }}>
              Smart Navigation
            </div>
            <p className="caption">Find the right doctor and specialty</p>
          </div>
        </div>

        {/* CTA */}
        <button
          className="btn btn-primary btn-lg"
          onClick={() => navigate('/language')}
          style={{ minWidth: 280, fontSize: 'var(--font-size-xl)' }}
        >
          Start →
        </button>

        <div style={{ marginTop: 'var(--space-6)' }}>
          <button className="btn btn-ghost btn-sm" onClick={() => {/* TODO: modal */}}>
            ❓ How it works
          </button>
        </div>

        {/* Value Prop */}
        <div style={{
          marginTop: 'var(--space-12)', padding: 'var(--space-5)',
          background: 'var(--color-surface)', borderRadius: 'var(--radius-xl)',
          border: '1px solid var(--color-border)', maxWidth: 560, margin: 'var(--space-12) auto 0'
        }}>
          <p className="caption" style={{ fontStyle: 'italic', lineHeight: 1.6 }}>
            "MediKiosk does not replace the doctor. It prepares the patient, the history,
            and the records before the consultation — so the doctor can focus on clinical decision-making."
          </p>
        </div>

      </div>
    </div>
  );
}
