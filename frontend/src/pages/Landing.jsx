import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

export default function Landing() {
  const navigate = useNavigate();
  const [showHowItWorks, setShowHowItWorks] = useState(false);

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
          <button className="btn btn-ghost btn-sm" onClick={() => setShowHowItWorks(true)}>
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

        {/* How It Works Modal */}
        {showHowItWorks && (
          <div style={{
            position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
            background: 'rgba(15, 23, 42, 0.6)', backdropFilter: 'blur(4px)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            zIndex: 1000, padding: 'var(--space-4)'
          }}>
            <div className="card animate-slide-up" style={{
              maxWidth: 580, width: '100%', maxHeight: '90vh', overflowY: 'auto',
              padding: 'var(--space-8)', textAlign: 'left', position: 'relative'
            }}>
              <div className="flex items-center justify-between mb-4">
                <h3 className="heading-3">How MediKiosk Works</h3>
                <button
                  onClick={() => setShowHowItWorks(false)}
                  style={{
                    background: 'none', border: 'none', fontSize: '1.5rem',
                    cursor: 'pointer', color: 'var(--color-text-muted)', lineHeight: 1
                  }}
                >
                  ✕
                </button>
              </div>

              <div className="flex flex-col gap-4 mb-6">
                <div style={{ display: 'flex', gap: 'var(--space-4)', alignItems: 'flex-start' }}>
                  <div style={{
                    width: 36, height: 36, borderRadius: '50%', background: 'var(--color-primary-50)',
                    color: 'var(--color-primary)', display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontWeight: 700, flexShrink: 0
                  }}>1</div>
                  <div>
                    <strong style={{ display: 'block', marginBottom: '2px' }}>Choose Language & Identify (ABHA)</strong>
                    <span className="caption">Select from 10 Indian languages and verify your Ayushman Bharat Health Account (ABHA).</span>
                  </div>
                </div>

                <div style={{ display: 'flex', gap: 'var(--space-4)', alignItems: 'flex-start' }}>
                  <div style={{
                    width: 36, height: 36, borderRadius: '50%', background: 'var(--color-primary-50)',
                    color: 'var(--color-primary)', display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontWeight: 700, flexShrink: 0
                  }}>2</div>
                  <div>
                    <strong style={{ display: 'block', marginBottom: '2px' }}>Voice-Driven Intake Consultation</strong>
                    <span className="caption">Speak naturally in your native language. Edge AI asks targeted follow-ups to gather HPI, PMH, allergies, and red flags.</span>
                  </div>
                </div>

                <div style={{ display: 'flex', gap: 'var(--space-4)', alignItems: 'flex-start' }}>
                  <div style={{
                    width: 36, height: 36, borderRadius: '50%', background: 'var(--color-primary-50)',
                    color: 'var(--color-primary)', display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontWeight: 700, flexShrink: 0
                  }}>3</div>
                  <div>
                    <strong style={{ display: 'block', marginBottom: '2px' }}>Smart OCR & Past Record Correlation</strong>
                    <span className="caption">Upload prescriptions or lab reports. AI automatically highlights past conditions relevant to today's complaint.</span>
                  </div>
                </div>

                <div style={{ display: 'flex', gap: 'var(--space-4)', alignItems: 'flex-start' }}>
                  <div style={{
                    width: 36, height: 36, borderRadius: '50%', background: 'var(--color-primary-50)',
                    color: 'var(--color-primary)', display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontWeight: 700, flexShrink: 0
                  }}>4</div>
                  <div>
                    <strong style={{ display: 'block', marginBottom: '2px' }}>Doctor Dashboard & OPD Handover</strong>
                    <span className="caption">The doctor receives a structured clinical summary with highlighted key findings before you even step in!</span>
                  </div>
                </div>
              </div>

              <button className="btn btn-primary btn-full" onClick={() => setShowHowItWorks(false)}>
                Got it!
              </button>
            </div>
          </div>
        )}

      </div>
    </div>
  );
}
