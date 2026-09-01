import React, { createContext, useContext, useState, useCallback } from 'react';
import { translations } from '../utils/translations';

const AppContext = createContext(null);

const LANGUAGES = [
  { code: 'en', label: 'English', native: 'English', speechCode: 'en-IN', flag: '🇬🇧' },
  { code: 'hi', label: 'Hindi', native: 'हिन्दी', speechCode: 'hi-IN', flag: '🇮🇳' },
  { code: 'mr', label: 'Marathi', native: 'मराठी', speechCode: 'mr-IN', flag: '🇮🇳' },
  { code: 'bn', label: 'Bengali', native: 'বাংলা', speechCode: 'bn-IN', flag: '🇮🇳' },
  { code: 'ta', label: 'Tamil', native: 'தமிழ்', speechCode: 'ta-IN', flag: '🇮🇳' },
  { code: 'te', label: 'Telugu', native: 'తెలుగు', speechCode: 'te-IN', flag: '🇮🇳' },
  { code: 'gu', label: 'Gujarati', native: 'ગુજરાતી', speechCode: 'gu-IN', flag: '🇮🇳' },
  { code: 'kn', label: 'Kannada', native: 'ಕನ್ನಡ', speechCode: 'kn-IN', flag: '🇮🇳' },
  { code: 'ml', label: 'Malayalam', native: 'മലയാളം', speechCode: 'ml-IN', flag: '🇮🇳' },
  { code: 'pa', label: 'Punjabi', native: 'ਪੰਜਾਬੀ', speechCode: 'pa-IN', flag: '🇮🇳' },
];

export const DEMO_ABHA_PATIENTS = [
  {
    name: 'Ramesh Sharma',
    age: 58,
    gender: 'Male',
    phone: '9876543210',
    abhaId: '12-3456-7890-1234',
    avatar: '👨‍🦳',
    badge: 'Cardiology & Diabetes',
    summary: 'Past STEMI (Heart Attack), Stented LAD, Type 2 DM, HTN'
  },
  {
    name: 'Priya Patel',
    age: 32,
    gender: 'Female',
    phone: '9812345678',
    abhaId: '23-4567-8901-2345',
    avatar: '👩',
    badge: 'Pulmonology & Asthma',
    summary: 'Chronic Bronchial Asthma, High IgE, Inhaler Therapy'
  },
  {
    name: 'Sunita Devi',
    age: 64,
    gender: 'Female',
    phone: '9765432109',
    abhaId: '34-5678-9012-3456',
    avatar: '👵',
    badge: 'Orthopedics & Arthritis',
    summary: 'Grade 3 Knee Osteoarthritis, Osteopenia, Bone Pain'
  },
  {
    name: 'Mohammed Ali',
    age: 45,
    gender: 'Male',
    phone: '9654321098',
    abhaId: '45-6789-0123-4567',
    avatar: '👨',
    badge: 'Gastroenterology & Liver',
    summary: 'Erosive GERD, Peptic Ulcer, Grade 1 Fatty Liver'
  },
  {
    name: 'Anita Verma',
    age: 26,
    gender: 'Female',
    phone: '9543210987',
    abhaId: '56-7890-1234-5678',
    avatar: '👩‍🦰',
    badge: 'ENT & Allergy',
    summary: 'Chronic Maxillary Sinusitis, Dust Mite Allergy'
  }
];

export function AppProvider({ children }) {
  // Language with localStorage persistence
  const [language, setLanguageState] = useState(() => {
    try {
      const saved = localStorage.getItem('medikiosk_language');
      if (saved) {
        const parsed = JSON.parse(saved);
        const code = parsed?.code || parsed;
        const matched = LANGUAGES.find(l => l.code === code);
        if (matched) return matched;
      }
    } catch (e) {
      // fallback
    }
    return LANGUAGES[0]; // Default to English if none selected
  });

  const setLanguage = useCallback((lang) => {
    if (!lang) {
      setLanguageState(LANGUAGES[0]);
      try { localStorage.removeItem('medikiosk_language'); } catch (e) {}
      return;
    }
    const target = typeof lang === 'string' 
      ? LANGUAGES.find(l => l.code === lang) || { code: lang, label: lang, native: lang, speechCode: `${lang}-IN` } 
      : lang;
    setLanguageState(target);
    try {
      localStorage.setItem('medikiosk_language', JSON.stringify(target));
    } catch (e) {}
  }, []);

  // Patient identity
  const [patient, setPatient] = useState(null);
  const [patientId, setPatientId] = useState(null); // backend-assigned PT-XXXX

  // Consent
  const [consentGiven, setConsentGiven] = useState(false);

  // Clinical mode
  const [clinicalMode, setClinicalMode] = useState(null); // 'allopathic' or 'ayush'

  // Clinical data (populated by Kiosk)
  const [clinicalData, setClinicalData] = useState(null);
  const [documents, setDocuments] = useState([]);
  const [redFlags, setRedFlags] = useState(null);
  const [specialty, setSpecialty] = useState(null);
  const [selectedProvider, setSelectedProvider] = useState(null);

  // Reset entire flow
  const resetFlow = useCallback(() => {
    setPatient(null);
    setPatientId(null);
    setConsentGiven(false);
    setClinicalMode(null);
    setClinicalData(null);
    setDocuments([]);
    setRedFlags(null);
    setSpecialty(null);
    setSelectedProvider(null);
  }, []);

  // Load demo patient (default Ramesh Sharma)
  const loadDemoPatient = useCallback((index = 0) => {
    const selected = DEMO_ABHA_PATIENTS[index] || DEMO_ABHA_PATIENTS[0];
    setPatient(selected);
  }, []);

  // Select specific ABHA patient
  const selectAbhaPatient = useCallback((patientObj) => {
    setPatient(patientObj);
  }, []);

  const languageCode = language?.code || 'en';

  const t = useCallback((key) => {
    const langDict = translations[languageCode] || translations['en'];
    return langDict?.[key] || translations['en']?.[key] || key;
  }, [languageCode]);

  const value = {
    // Language
    language, setLanguage,
    languages: LANGUAGES,

    // Patient
    patient, setPatient,
    patientId, setPatientId,
    loadDemoPatient,
    selectAbhaPatient,
    demoAbhaPatients: DEMO_ABHA_PATIENTS,

    // Consent
    consentGiven, setConsentGiven,

    // Clinical
    clinicalMode, setClinicalMode,
    clinicalData, setClinicalData,
    documents, setDocuments,
    redFlags, setRedFlags,
    specialty, setSpecialty,
    selectedProvider, setSelectedProvider,

    // Utils
    resetFlow,
    isAyush: clinicalMode === 'ayush',
    languageLabel: language?.label || 'English',
    languageNative: language?.native || 'English',
    languageCode,
    speechCode: language?.speechCode || 'en-IN',
    t,
  };

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

export function useApp() {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error('useApp must be used within AppProvider');
  return ctx;
}

export default AppContext;
