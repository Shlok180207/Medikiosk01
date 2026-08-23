import React, { createContext, useContext, useState, useCallback } from 'react';
import { translations } from '../utils/translations';

const AppContext = createContext(null);

const LANGUAGES = [
  { code: 'en', label: 'English', native: 'English', speechCode: 'en-IN' },
  { code: 'hi', label: 'Hindi', native: 'हिन्दी', speechCode: 'hi-IN' },
  { code: 'mr', label: 'Marathi', native: 'मराठी', speechCode: 'mr-IN' },
  { code: 'bn', label: 'Bengali', native: 'বাংলা', speechCode: 'bn-IN' },
  { code: 'ta', label: 'Tamil', native: 'தமிழ்', speechCode: 'ta-IN' },
  { code: 'te', label: 'Telugu', native: 'తెలుగు', speechCode: 'te-IN' },
  { code: 'gu', label: 'Gujarati', native: 'ગુજરાતી', speechCode: 'gu-IN' },
  { code: 'kn', label: 'Kannada', native: 'ಕನ್ನಡ', speechCode: 'kn-IN' },
  { code: 'ml', label: 'Malayalam', native: 'മലയാളം', speechCode: 'ml-IN' },
  { code: 'pa', label: 'Punjabi', native: 'ਪੰਜਾਬੀ', speechCode: 'pa-IN' },
];

const DEMO_PATIENT = {
  name: 'Demo Patient',
  age: 48,
  gender: 'Male',
  abhaId: 'ABHA-DEMO-1234-5678',
  phone: '9876543210',
};

export function AppProvider({ children }) {
  // Language
  const [language, setLanguage] = useState(null); // { code, label, native, speechCode }

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
    setLanguage(null);
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

  // Load demo patient
  const loadDemoPatient = useCallback(() => {
    setPatient(DEMO_PATIENT);
  }, []);

  const languageCode = language?.code || 'en';

  const t = useCallback((key) => {
    const langDict = translations[languageCode] || translations['en'];
    return langDict[key] || translations['en'][key] || key;
  }, [languageCode]);

  const value = {
    // Language
    language, setLanguage,
    languages: LANGUAGES,

    // Patient
    patient, setPatient,
    patientId, setPatientId,
    loadDemoPatient,

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
