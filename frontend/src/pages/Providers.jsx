import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useApp } from '../context/AppContext';

// Dummy doctors from "City District Hospital" — 2 per specialty
const HOSPITAL_NAME = 'City District Hospital';
const HOSPITAL_DOCTORS = [
  // General Medicine
  { id: 1, name: 'Dr. Rahul Sharma', specialty: 'General Medicine', qualifications: 'MBBS, MD (Internal Medicine)', room: 'OPD Room 101', availability: 'Available', timing: 'Mon–Sat: 9:00 AM – 1:00 PM', photo: '👨‍⚕️' },
  { id: 2, name: 'Dr. Meena Gupta', specialty: 'General Medicine', qualifications: 'MBBS, DNB (General Medicine)', room: 'OPD Room 102', availability: 'Available', timing: 'Mon–Fri: 2:00 PM – 5:00 PM', photo: '👩‍⚕️' },

  // Cardiology
  { id: 3, name: 'Dr. Vikram Patel', specialty: 'Cardiology', qualifications: 'MBBS, MD, DM (Cardiology)', room: 'OPD Room 201', availability: 'Available', timing: 'Mon–Sat: 10:00 AM – 1:00 PM', photo: '👨‍⚕️' },
  { id: 4, name: 'Dr. Sunita Reddy', specialty: 'Cardiology', qualifications: 'MBBS, MD (Medicine), DNB (Cardiology)', room: 'OPD Room 202', availability: 'By Appointment', timing: 'Tue, Thu: 3:00 PM – 6:00 PM', photo: '👩‍⚕️' },

  // Orthopedics
  { id: 5, name: 'Dr. Arun Singh', specialty: 'Orthopedics', qualifications: 'MBBS, MS (Orthopedics)', room: 'OPD Room 301', availability: 'Available', timing: 'Mon–Sat: 9:30 AM – 12:30 PM', photo: '👨‍⚕️' },
  { id: 6, name: 'Dr. Kavita Joshi', specialty: 'Orthopedics', qualifications: 'MBBS, DNB (Orthopedics)', room: 'OPD Room 302', availability: 'Available', timing: 'Mon–Fri: 2:00 PM – 4:30 PM', photo: '👩‍⚕️' },

  // Gastroenterology
  { id: 7, name: 'Dr. Ramesh Iyer', specialty: 'Gastroenterology', qualifications: 'MBBS, MD, DM (Gastroenterology)', room: 'OPD Room 401', availability: 'Available', timing: 'Mon–Sat: 10:00 AM – 1:00 PM', photo: '👨‍⚕️' },
  { id: 8, name: 'Dr. Neha Kapoor', specialty: 'Gastroenterology', qualifications: 'MBBS, MD (Medicine), DNB (Gastro)', room: 'OPD Room 402', availability: 'By Appointment', timing: 'Wed, Fri: 3:00 PM – 6:00 PM', photo: '👩‍⚕️' },

  // Pulmonology
  { id: 9, name: 'Dr. Sanjay Mishra', specialty: 'Pulmonology', qualifications: 'MBBS, MD (Pulmonary Medicine)', room: 'OPD Room 501', availability: 'Available', timing: 'Mon–Sat: 9:00 AM – 12:00 PM', photo: '👨‍⚕️' },
  { id: 10, name: 'Dr. Anita Das', specialty: 'Pulmonology', qualifications: 'MBBS, DNB (Respiratory Medicine)', room: 'OPD Room 502', availability: 'Available', timing: 'Mon–Fri: 2:30 PM – 5:00 PM', photo: '👩‍⚕️' },

  // Neurology
  { id: 11, name: 'Dr. Deepak Nair', specialty: 'Neurology', qualifications: 'MBBS, MD, DM (Neurology)', room: 'OPD Room 601', availability: 'By Appointment', timing: 'Mon, Wed, Fri: 10:00 AM – 1:00 PM', photo: '👨‍⚕️' },
  { id: 12, name: 'Dr. Pooja Tiwari', specialty: 'Neurology', qualifications: 'MBBS, MD (Medicine), DNB (Neuro)', room: 'OPD Room 602', availability: 'Available', timing: 'Tue, Thu: 2:00 PM – 5:00 PM', photo: '👩‍⚕️' },

  // Dermatology
  { id: 13, name: 'Dr. Amit Saxena', specialty: 'Dermatology', qualifications: 'MBBS, MD (Dermatology)', room: 'OPD Room 701', availability: 'Available', timing: 'Mon–Sat: 10:30 AM – 1:30 PM', photo: '👨‍⚕️' },
  { id: 14, name: 'Dr. Ritu Verma', specialty: 'Dermatology', qualifications: 'MBBS, DVD, DNB (Dermatology)', room: 'OPD Room 702', availability: 'Available', timing: 'Mon–Fri: 3:00 PM – 5:30 PM', photo: '👩‍⚕️' },

  // ENT
  { id: 15, name: 'Dr. Suresh Rao', specialty: 'ENT', qualifications: 'MBBS, MS (ENT)', room: 'OPD Room 801', availability: 'Available', timing: 'Mon–Sat: 9:00 AM – 12:00 PM', photo: '👨‍⚕️' },
  { id: 16, name: 'Dr. Lata Menon', specialty: 'ENT', qualifications: 'MBBS, DNB (ENT)', room: 'OPD Room 802', availability: 'By Appointment', timing: 'Tue, Thu, Sat: 2:00 PM – 4:30 PM', photo: '👩‍⚕️' },

  // Ophthalmology
  { id: 17, name: 'Dr. Rajiv Kumar', specialty: 'Ophthalmology', qualifications: 'MBBS, MS (Ophthalmology)', room: 'OPD Room 901', availability: 'Available', timing: 'Mon–Sat: 10:00 AM – 1:00 PM', photo: '👨‍⚕️' },
  { id: 18, name: 'Dr. Swati Bose', specialty: 'Ophthalmology', qualifications: 'MBBS, DNB (Ophthalmology)', room: 'OPD Room 902', availability: 'Available', timing: 'Mon–Fri: 2:00 PM – 5:00 PM', photo: '👩‍⚕️' },

  // Pediatrics
  { id: 19, name: 'Dr. Manoj Pandey', specialty: 'Pediatrics', qualifications: 'MBBS, MD (Pediatrics)', room: 'OPD Room 1001', availability: 'Available', timing: 'Mon–Sat: 9:00 AM – 1:00 PM', photo: '👨‍⚕️' },
  { id: 20, name: 'Dr. Nisha Agarwal', specialty: 'Pediatrics', qualifications: 'MBBS, DCH, DNB (Pediatrics)', room: 'OPD Room 1002', availability: 'Available', timing: 'Mon–Fri: 3:00 PM – 6:00 PM', photo: '👩‍⚕️' },

  // Obstetrics & Gynecology
  { id: 21, name: 'Dr. Shalini Yadav', specialty: 'Obstetrics & Gynecology', qualifications: 'MBBS, MS (OBG)', room: 'OPD Room 1101', availability: 'Available', timing: 'Mon–Sat: 9:30 AM – 12:30 PM', photo: '👩‍⚕️' },
  { id: 22, name: 'Dr. Geeta Prasad', specialty: 'Obstetrics & Gynecology', qualifications: 'MBBS, DGO, DNB (OBG)', room: 'OPD Room 1102', availability: 'Available', timing: 'Mon–Fri: 2:00 PM – 5:00 PM', photo: '👩‍⚕️' },

  // Psychiatry
  { id: 23, name: 'Dr. Ashok Menon', specialty: 'Psychiatry', qualifications: 'MBBS, MD (Psychiatry)', room: 'OPD Room 1201', availability: 'By Appointment', timing: 'Mon, Wed, Fri: 10:00 AM – 1:00 PM', photo: '👨‍⚕️' },
  { id: 24, name: 'Dr. Prerna Sinha', specialty: 'Psychiatry', qualifications: 'MBBS, DNB (Psychiatry)', room: 'OPD Room 1202', availability: 'By Appointment', timing: 'Tue, Thu: 2:00 PM – 5:00 PM', photo: '👩‍⚕️' },

  // Urology
  { id: 25, name: 'Dr. Harish Chandra', specialty: 'Urology', qualifications: 'MBBS, MS, MCh (Urology)', room: 'OPD Room 1301', availability: 'By Appointment', timing: 'Mon, Wed, Fri: 10:00 AM – 12:00 PM', photo: '👨‍⚕️' },
  { id: 26, name: 'Dr. Kiran Desai', specialty: 'Urology', qualifications: 'MBBS, DNB (Urology)', room: 'OPD Room 1302', availability: 'Available', timing: 'Tue, Thu, Sat: 2:00 PM – 4:30 PM', photo: '👨‍⚕️' },

  // Surgery
  { id: 27, name: 'Dr. Prakash Jha', specialty: 'Surgery', qualifications: 'MBBS, MS (General Surgery)', room: 'OPD Room 1401', availability: 'Available', timing: 'Mon–Sat: 9:00 AM – 12:00 PM', photo: '👨‍⚕️' },
  { id: 28, name: 'Dr. Shobha Nambiar', specialty: 'Surgery', qualifications: 'MBBS, DNB (General Surgery)', room: 'OPD Room 1402', availability: 'By Appointment', timing: 'Tue, Thu: 2:00 PM – 5:00 PM', photo: '👩‍⚕️' },
];

export default function Providers() {
  const navigate = useNavigate();
  const { specialty, setSelectedProvider, t } = useApp();
  const [selected, setSelected] = useState(null);

  const matchedSpecialty = specialty?.specialty || 'General Medicine';

  // Filter doctors by the AI-recommended specialty
  const matchedDoctors = HOSPITAL_DOCTORS.filter(d => d.specialty === matchedSpecialty);

  // If no exact match found, fall back to General Medicine
  const displayDoctors = matchedDoctors.length > 0
    ? matchedDoctors
    : HOSPITAL_DOCTORS.filter(d => d.specialty === 'General Medicine');

  const handleSelect = (doctor) => {
    setSelected(doctor.id);
    setSelectedProvider({
      ...doctor,
      facility: HOSPITAL_NAME,
    });
  };

  const handleContinue = () => {
    navigate('/summary');
  };

  return (
    <div className="page-container animate-fade-in" style={{ justifyContent: 'flex-start', paddingTop: 'var(--space-8)' }}>
      <div className="page-content-wide">

        {/* Header */}
        <div className="text-center mb-6">
          <h2 className="heading-2 mb-2">👨‍⚕️ Select Your Doctor</h2>
          <p className="subtitle">
            Based on your diagnosis, we recommend a <strong>{matchedSpecialty}</strong> specialist
          </p>
          <p className="caption mt-2">🏥 {HOSPITAL_NAME}</p>
        </div>

        {/* Info Banner */}
        <div className="alert alert-info mb-6">
          🤖 <strong>AI-Recommended Specialty:</strong> {matchedSpecialty}
          {specialty?.reason && <span> — {specialty.reason}</span>}
        </div>

        {/* Doctor Cards */}
        <div className="flex flex-col gap-4 mb-6">
          {displayDoctors.map(doctor => (
            <div
              key={doctor.id}
              className={`card card-interactive ${selected === doctor.id ? 'active' : ''}`}
              onClick={() => handleSelect(doctor)}
              style={{ padding: 'var(--space-6)', cursor: 'pointer' }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 'var(--space-4)' }}>
                <div style={{ flex: 1 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-3)', marginBottom: 'var(--space-2)' }}>
                    <span style={{ fontSize: '2rem' }}>{doctor.photo}</span>
                    <div>
                      <h3 className="heading-4" style={{ marginBottom: 'var(--space-1)' }}>{doctor.name}</h3>
                      <p className="caption" style={{ color: 'var(--color-primary)', fontWeight: 600 }}>{doctor.specialty}</p>
                    </div>
                  </div>
                  <p className="caption" style={{ marginBottom: 'var(--space-1)' }}>{doctor.qualifications}</p>
                  <div className="flex items-center gap-4 mt-2" style={{ flexWrap: 'wrap' }}>
                    <span className="caption">📍 {doctor.room}</span>
                    <span className="caption">🕐 {doctor.timing}</span>
                  </div>
                </div>
                <div style={{ textAlign: 'right' }}>
                  <div style={{
                    display: 'inline-flex', alignItems: 'center', gap: 'var(--space-2)',
                    padding: 'var(--space-1) var(--space-3)', borderRadius: 'var(--radius-full)',
                    background: doctor.availability === 'Available' ? 'var(--color-success-bg)' : 'var(--color-warning-bg)',
                    color: doctor.availability === 'Available' ? '#059669' : '#d97706',
                    fontSize: 'var(--font-size-sm)', fontWeight: 600
                  }}>
                    <span style={{ width: 8, height: 8, borderRadius: '50%', background: doctor.availability === 'Available' ? '#059669' : '#d97706' }} />
                    {doctor.availability}
                  </div>
                </div>
              </div>

              {selected === doctor.id && (
                <div className="flex gap-3 mt-4" style={{ borderTop: '1px solid var(--color-border)', paddingTop: 'var(--space-4)' }}>
                  <button className="btn btn-primary btn-sm" style={{ flex: 1 }} onClick={handleContinue}>
                    ✓ Confirm & Continue
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
