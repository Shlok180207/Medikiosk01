import React, { useState, useEffect } from 'react';

const API_BASE_URL = 'http://localhost:8000/api';

export default function Dashboard() {
  const [queue, setQueue] = useState([]);
  const [selectedPatientId, setSelectedPatientId] = useState(null);
  const [patientData, setPatientData] = useState(null);
  const [status, setStatus] = useState('Waiting for patient...');

  // Fetch queue
  const fetchQueue = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/patients`);
      const patients = await response.json();
      setQueue(patients);
    } catch (error) {
      console.error('Failed to fetch queue', error);
    }
  };

  // Auto-select first patient only if none selected
  useEffect(() => {
    if (queue.length > 0 && !selectedPatientId) {
      setSelectedPatientId(queue[0].patient_id);
    }
  }, [queue, selectedPatientId]);

  // Fetch patient data
  const fetchData = async () => {
    if (!selectedPatientId) return;
    try {
      setStatus('Fetching...');
      const response = await fetch(`${API_BASE_URL}/patient-summary?patient_id=${selectedPatientId}`);
      const data = await response.json();
      setPatientData(data);
      setStatus('Synchronized');
    } catch (error) {
      console.error('Failed to fetch data', error);
      setStatus('Error connecting');
    }
  };

  // Delete patient
  const handleDelete = async (e, id) => {
    e.stopPropagation();
    if (!window.confirm(`Are you sure you want to delete patient ${id}?`)) return;
    try {
      await fetch(`${API_BASE_URL}/patients/${id}`, { method: 'DELETE' });
      if (selectedPatientId === id) {
        setSelectedPatientId(null);
        setPatientData(null);
      }
      fetchQueue();
    } catch (error) {
      console.error('Failed to delete patient', error);
    }
  };

  useEffect(() => {
    const interval = setInterval(fetchQueue, 5000);
    fetchQueue();
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (selectedPatientId) fetchData();
  }, [selectedPatientId]);

  const isEmergency = patientData?.is_emergency;

  return (
    <div style={{ display: 'flex', height: '100vh', background: 'var(--color-bg)' }}>

      {/* Sidebar — Patient Queue */}
      <aside style={{
        width: 300, flexShrink: 0, background: 'var(--color-surface)',
        borderRight: '1px solid var(--color-border)', display: 'flex', flexDirection: 'column'
      }}>
        <div style={{ padding: 'var(--space-5)', borderBottom: '1px solid var(--color-border)' }}>
          <h3 style={{ fontSize: 'var(--font-size-lg)', fontWeight: 700 }}>🏥 Patient Queue</h3>
          <p className="caption">{queue.length} patient(s)</p>
        </div>
        <div style={{ flex: 1, overflowY: 'auto' }}>
          {queue.length === 0 ? (
            <div style={{ padding: 'var(--space-6)', textAlign: 'center' }}>
              <p className="caption">No patients in queue</p>
            </div>
          ) : (
            queue.map(p => (
              <div
                key={p.patient_id}
                onClick={() => setSelectedPatientId(p.patient_id)}
                style={{
                  padding: 'var(--space-4) var(--space-5)',
                  borderBottom: '1px solid var(--color-border-light)',
                  cursor: 'pointer',
                  background: selectedPatientId === p.patient_id ? 'var(--color-primary-50)' : 'transparent',
                  borderLeft: selectedPatientId === p.patient_id ? '3px solid var(--color-primary)' : '3px solid transparent',
                  transition: 'all var(--transition-fast)'
                }}
              >
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-2">
                    {p.is_emergency && <span style={{ color: 'var(--color-danger)' }}>🚨</span>}
                    <span style={{ fontWeight: 600, color: 'var(--color-text)' }}>{p.patient_id}</span>
                  </div>
                  <button 
                    onClick={(e) => handleDelete(e, p.patient_id)} 
                    style={{ background: 'none', border: 'none', cursor: 'pointer', opacity: 0.6 }}
                    title="Delete patient"
                  >
                    🗑️
                  </button>
                </div>
                <span className="caption">Arrived: {p.created_at}</span>
              </div>
            ))
          )}
        </div>
      </aside>

      {/* Main Content */}
      <main style={{ flex: 1, overflowY: 'auto', padding: 'var(--space-6)' }}>

        {!patientData ? (
          <div className="page-container" style={{ minHeight: 'auto' }}>
            <div className="text-center">
              <div style={{ fontSize: '3rem', marginBottom: 'var(--space-4)' }}>👨‍⚕️</div>
              <h2 className="heading-2 mb-2">MediKiosk Clinical Dashboard</h2>
              <p className="subtitle">Select a patient from the queue to view their clinical summary</p>
            </div>
          </div>
        ) : (
          <div className="animate-fade-in">
            {/* Header */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-6)', flexWrap: 'wrap', gap: 'var(--space-4)' }}>
              <div>
                <h2 className="heading-2">Clinical Summary</h2>
                <p className="caption">
                  AI-Generated Intake Report • {patientData.patient_id} • {status}
                </p>
              </div>
              <div className="flex gap-3">
                <button className="btn btn-secondary btn-sm" onClick={fetchData}>🔄 Refresh</button>
                <button className="btn btn-primary btn-sm">✓ Start Consultation</button>
              </div>
            </div>

            {/* Emergency Banner */}
            {isEmergency && (
              <div className="alert alert-danger mb-6" style={{ fontSize: 'var(--font-size-lg)', fontWeight: 700 }}>
                🚨 URGENT MEDICAL ALERT — High Severity Symptoms Detected!
              </div>
            )}

            {/* Draft Banner */}
            <div className="alert alert-warning mb-6">
              ⚠️ AI-GENERATED DRAFT — VERIFY BEFORE CLINICAL USE
            </div>

            {/* Cards Grid */}
            <div className="grid-2" style={{ gap: 'var(--space-4)', alignItems: 'start' }}>

              {/* Left */}
              <div className="flex flex-col gap-4">
                <DashCard title="🎯 Chief Complaint" content={patientData.chief_complaint}
                  highlight severity={patientData.severity} />
                <DashCard title="📋 History of Present Illness" content={patientData.hpi} />
                <DashCard title="📂 Past Medical History" content={patientData.past_medical_history} />
                <DashCard title="👨‍👩‍👧‍👦 Family History" content={patientData.family_history} />
              </div>

              {/* Right */}
              <div className="flex flex-col gap-4">
                <DashCard title="🏃 Personal / Lifestyle" content={patientData.personal_history} />
                <DashCard title="⚠️ Allergies" content={patientData.allergies} />
                <DashCard title="🔍 Review of Systems" content={patientData.review_of_systems} />

                {/* Quick Stats */}
                <div className="card" style={{ padding: 'var(--space-5)' }}>
                  <h4 style={{ fontWeight: 600, marginBottom: 'var(--space-3)' }}>📊 Assessment</h4>
                  <div className="flex flex-col gap-2">
                    <StatRow label="Severity" value={patientData.severity}
                      badge={patientData.severity === 'High' ? 'badge-danger' : patientData.severity === 'Medium' ? 'badge-warning' : 'badge-success'} />
                    <StatRow label="Duration" value={patientData.duration} />
                    <StatRow label="Emergency"
                      value={patientData.is_emergency ? 'Yes' : 'No'}
                      badge={patientData.is_emergency ? 'badge-danger' : 'badge-success'} />
                  </div>
                </div>

                {/* Documents */}
                {patientData.flagged_lab_values && patientData.flagged_lab_values !== '[]' && (
                  <div className="card" style={{ padding: 'var(--space-5)' }}>
                    <h4 style={{ fontWeight: 600, marginBottom: 'var(--space-3)' }}>📄 Processed Documents</h4>
                    <DocumentsView data={patientData.flagged_lab_values} patientId={patientData.patient_id} />
                  </div>
                )}
              </div>
            </div>

            {/* AYUSH */}
            {patientData.prakriti && patientData.prakriti !== 'Not assessed' && (
              <div className="card mt-4" style={{ padding: 'var(--space-5)', borderLeft: '4px solid var(--color-success)' }}>
                <h4 style={{ fontWeight: 600, marginBottom: 'var(--space-3)' }}>🌿 AYUSH Assessment</h4>
                <div className="grid-3">
                  <div><span className="caption">Prakriti</span><p className="body-text">{patientData.prakriti}</p></div>
                  <div><span className="caption">Vikriti</span><p className="body-text">{patientData.vikriti}</p></div>
                  <div><span className="caption">Agni</span><p className="body-text">{patientData.agni}</p></div>
                </div>
              </div>
            )}

          </div>
        )}
      </main>
    </div>
  );
}

function DashCard({ title, content, highlight, severity }) {
  const isEmpty = !content || content === 'None reported' || content === 'Not recorded';
  return (
    <div className="card" style={{
      padding: 'var(--space-5)',
      borderLeft: highlight ? `4px solid ${severity === 'High' ? 'var(--color-danger)' : 'var(--color-primary)'}` : undefined,
      opacity: isEmpty ? 0.5 : 1
    }}>
      <h4 style={{ fontWeight: 600, marginBottom: 'var(--space-2)', fontSize: 'var(--font-size-base)' }}>{title}</h4>
      <p className="body-text" style={{ whiteSpace: 'pre-wrap' }}>
        {isEmpty ? 'Not reported' : content}
      </p>
    </div>
  );
}

function StatRow({ label, value, badge }) {
  return (
    <div className="flex items-center gap-2">
      <span className="caption" style={{ width: 90 }}>{label}:</span>
      {badge ? (
        <span className={`badge ${badge}`}>{value || 'Unknown'}</span>
      ) : (
        <span className="body-text">{value || 'Unknown'}</span>
      )}
    </div>
  );
}

function DocumentsView({ data, patientId }) {
  try {
    const parsed = JSON.parse(data);
    if (!Array.isArray(parsed) || parsed.length === 0) return <p className="caption">No documents</p>;
    return parsed.map((doc, i) => (
      <div key={i} style={{ marginBottom: 'var(--space-4)', paddingBottom: 'var(--space-4)', borderBottom: i < parsed.length - 1 ? '1px solid var(--color-border)' : 'none' }}>
        <h5 style={{ fontWeight: 700, marginBottom: 'var(--space-2)', fontSize: 'var(--font-size-base)', color: 'var(--color-primary)' }}>
          📄 {doc.document_type || 'Medical Document'}
        </h5>
        
        <div className="flex items-center gap-4 mb-3">
           <span className="caption">👤 Patient: {patientId}</span>
           <span className="caption">📅 Date: {doc.document_date || 'Unknown'}</span>
        </div>
        
        {doc.summary && (
          <div className="mb-3">
            <span className="caption" style={{ fontWeight: 600, display: 'block' }}>Summary:</span>
            <p className="body-text">{doc.summary}</p>
          </div>
        )}
        
        {doc.diagnoses?.length > 0 && <p className="caption mb-1"><strong>Diagnoses:</strong> {doc.diagnoses.join(', ')}</p>}
        {doc.medications?.length > 0 && <p className="caption mb-1"><strong>Medications:</strong> {doc.medications.join(', ')}</p>}
        {doc.flagged_values?.length > 0 && (
          <p className="caption mt-2" style={{ color: 'var(--color-danger)', fontWeight: 600 }}>
            ⚠️ Flagged Findings: {doc.flagged_values.join(', ')}
          </p>
        )}
        
        {doc.file_url && (
           <a href={doc.file_url} target="_blank" rel="noreferrer" className="btn btn-outline btn-sm mt-3" style={{ display: 'inline-flex', alignItems: 'center', gap: 'var(--space-2)' }}>
             👁️ View Full Original Document
           </a>
        )}
      </div>
    ));
  } catch {
    return <p className="caption">{data}</p>;
  }
}
