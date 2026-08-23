import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { AppProvider } from './context/AppContext';
import Landing from './pages/Landing';
import LanguageSelect from './pages/LanguageSelect';
import PatientID from './pages/PatientID';
import Consent from './pages/Consent';
import ClinicalMode from './pages/ClinicalMode';
import Kiosk from './pages/Kiosk';
import DocumentScan from './pages/DocumentScan';
import RedFlag from './pages/RedFlag';
import Specialty from './pages/Specialty';
import Providers from './pages/Providers';
import Summary from './pages/Summary';
import PatientFinal from './pages/PatientFinal';
import Dashboard from './pages/Dashboard';
import './index.css';

function App() {
  return (
    <AppProvider>
      <Router>
        <Routes>
          {/* Patient Flow */}
          <Route path="/" element={<Landing />} />
          <Route path="/language" element={<LanguageSelect />} />
          <Route path="/patient-id" element={<PatientID />} />
          <Route path="/consent" element={<Consent />} />
          <Route path="/clinical-mode" element={<ClinicalMode />} />
          <Route path="/kiosk" element={<Kiosk />} />
          <Route path="/document-scan" element={<DocumentScan />} />
          <Route path="/red-flag" element={<RedFlag />} />
          <Route path="/specialty" element={<Specialty />} />
          <Route path="/providers" element={<Providers />} />
          <Route path="/summary" element={<Summary />} />
          <Route path="/patient-final" element={<PatientFinal />} />

          {/* Doctor */}
          <Route path="/doctor" element={<Dashboard />} />
        </Routes>
      </Router>
    </AppProvider>
  );
}

export default App;
