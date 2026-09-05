import React, { useState, useRef, useCallback, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useApp } from '../context/AppContext';

import { API_BASE_URL } from '../config';

export default function Kiosk() {
  const navigate = useNavigate();
  const { languageLabel, languageCode, speechCode, isAyush, patient, patientId, setPatientId, setClinicalData, t } = useApp();

  // ── TTS (Text-to-Speech) for accessibility ──
  const audioRef = useRef(null);
  const ttsUnlockedRef = useRef(false);
  const pendingSpeechRef = useRef(null);

  const stopSpeaking = useCallback(() => {
    pendingSpeechRef.current = null;
    ttsUnlockedRef.current = true;
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
      audioRef.current = null;
    }
    if (window.speechSynthesis) {
      window.speechSynthesis.cancel();
    }
  }, []);

  const speakText = useCallback((text) => {
    if (!text) return;
    const skip = ['Processing...', 'Listening...', 'Noted.', 'Skipping...'];
    if (skip.some(s => text.startsWith(s))) return;

    try {
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current = null;
      }
      const lang = languageLabel || 'Hindi';
      const encoded = encodeURIComponent(text);
      const audio = new Audio(`${API_BASE_URL}/tts?text=${encoded}&lang=${lang}`);
      audioRef.current = audio;

      const playPromise = audio.play();
      if (playPromise !== undefined) {
        playPromise
          .then(() => {
            ttsUnlockedRef.current = true;
          })
          .catch(err => {
            console.warn('Audio play error, falling back to Web Speech API:', err);
            if (window.speechSynthesis) {
              window.speechSynthesis.cancel();
              const utterance = new SpeechSynthesisUtterance(text);
              utterance.lang = speechCode || 'hi-IN';
              utterance.rate = 0.9;
              window.speechSynthesis.speak(utterance);
            }
          });
      }
    } catch (err) {
      console.error('TTS error:', err);
    }
  }, [languageLabel, speechCode]);

  // Unlock audio on first user click or tap
  useEffect(() => {
    pendingSpeechRef.current = getGreeting(languageLabel);

    const unlockTTS = () => {
      if (!ttsUnlockedRef.current && pendingSpeechRef.current) {
        speakText(pendingSpeechRef.current);
        pendingSpeechRef.current = null;
      }
      document.removeEventListener('click', unlockTTS);
      document.removeEventListener('touchstart', unlockTTS);
    };

    document.addEventListener('click', unlockTTS);
    document.addEventListener('touchstart', unlockTTS);

    return () => {
      document.removeEventListener('click', unlockTTS);
      document.removeEventListener('touchstart', unlockTTS);
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current = null;
      }
      window.speechSynthesis?.cancel();
    };
  }, [languageLabel, speakText]);

  // Guard: Redirect to ABHA verification if no patient is authenticated
  useEffect(() => {
    if (!patient || !patient.name) {
      console.warn('No authenticated ABHA patient found. Redirecting to /patient-id...');
      navigate('/patient-id');
    }
  }, [patient, navigate]);

  // Chat state
  const [messages, setMessages] = useState([
    { type: 'bot', text: getGreeting(languageLabel) }
  ]);
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [conversationPhase, setConversationPhase] = useState('initial'); // initial, follow-up, complete
  const [conversationContext, setConversationContext] = useState('');
  const [followUpCount, setFollowUpCount] = useState(0);
  const [progress, setProgress] = useState(10);
  const [textInput, setTextInput] = useState('');

  // Refs
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const chatEndRef = useRef(null);
  const recognitionRef = useRef(null);

  // Auto-scroll to latest message
  const scrollToBottom = useCallback(() => {
    setTimeout(() => chatEndRef.current?.scrollIntoView({ behavior: 'smooth' }), 100);
  }, []);

  const addMessage = useCallback((text, type) => {
    setMessages(prev => [...prev, { type, text }]);
    scrollToBottom();
    // Auto-speak bot messages
    if (type === 'bot') {
      setTimeout(() => speakText(text), 200);
    }
  }, [scrollToBottom, speakText]);

  const removeLastBotMessage = useCallback(() => {
    setMessages(prev => {
      const last = [...prev];
      for (let i = last.length - 1; i >= 0; i--) {
        if (last[i].type === 'bot' && (last[i].text === 'Processing...' || last[i].text === 'Listening...')) {
          last.splice(i, 1);
          break;
        }
      }
      return last;
    });
  }, []);

  // ── Voice Recording (Whisper offline) ──
  const handleMicClick = async () => {
    stopSpeaking();
    if (isRecording) {
      if (recognitionRef.current) recognitionRef.current.stop();
      if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
        mediaRecorderRef.current.stop();
        mediaRecorderRef.current.stream.getTracks().forEach(t => t.stop());
      }
      setIsRecording(false);
      return;
    }
    // Always use Whisper (offline) for maximum reliability
    await startWhisperRecording();
  };

  const startWhisperRecording = async () => {
    stopSpeaking();
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) audioChunksRef.current.push(event.data);
      };

      mediaRecorder.onstop = async () => {
        stream.getTracks().forEach(t => t.stop());
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        audioChunksRef.current = [];
        addMessage('🎤 Voice recorded — processing...', 'user');
        addMessage('Processing...', 'bot');
        setIsProcessing(true);
        if (conversationPhase === 'initial') await sendInitialAudio(audioBlob);
        else if (conversationPhase === 'follow-up') await sendFollowUpAudio(audioBlob);
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (err) {
      addMessage('⚠️ Microphone access denied. Please allow access and try again.', 'bot');
    }
  };

  // ── Text Input Handler ──
  const handleTextSubmit = async () => {
    stopSpeaking();
    if (!textInput.trim() || isProcessing) return;
    const text = textInput.trim();
    setTextInput('');
    addMessage(text, 'user');
    addMessage('Processing...', 'bot');
    setIsProcessing(true);
    if (conversationPhase === 'initial') await sendInitialText(text);
    else if (conversationPhase === 'follow-up') await sendFollowUpText(text);
  };

  // ── Quick Touch Answers ──
  const handleQuickAnswer = async (answer) => {
    stopSpeaking();
    if (isProcessing) return;
    addMessage(answer, 'user');
    addMessage('Processing...', 'bot');
    setIsProcessing(true);
    if (conversationPhase === 'initial') await sendInitialText(answer);
    else if (conversationPhase === 'follow-up') await sendFollowUpText(answer);
  };

  // ── Skip (Demo) ──
  const handleSkipDemo = async () => {
    stopSpeaking();
    if (isProcessing) return;
    setIsProcessing(true);
    addMessage('Skipping... generating demo patient and lab report...', 'bot');
    try {
      const selectedAbha = patient?.abhaId || '12-3456-7890-1234';
      const response = await fetch(`${API_BASE_URL}/demo-data?abha_id=${encodeURIComponent(selectedAbha)}`, { method: 'POST' });
      const data = await response.json();
      if (data.patient_id) {
        setPatientId(data.patient_id);
        setTimeout(() => navigate('/specialty'), 1000);
      }
    } catch (error) {
      console.error(error);
      removeLastBotMessage();
      addMessage('Error generating demo data', 'bot');
      setIsProcessing(false);
    }
  };

  // ── API Calls ──
  const sendInitialAudio = async (audioBlob) => {
    try {
      const formData = new FormData();
      formData.append('audio', audioBlob, 'recording.webm');
      formData.append('language', languageLabel);
      formData.append('is_ayush', isAyush);
      if (patient?.abhaId) formData.append('abha_id', patient.abhaId);
      if (patient?.name) formData.append('patient_name', patient.name);
      if (patient?.age) formData.append('age', patient.age);
      if (patient?.gender) formData.append('gender', patient.gender);
      if (patient?.phone) formData.append('phone', patient.phone);

      const response = await fetch(`${API_BASE_URL}/process-audio`, { method: 'POST', body: formData });
      const data = await response.json();
      handleInitialResponse(data);
    } catch (error) {
      removeLastBotMessage();
      addMessage('Sorry, there was an error. Please try again.', 'bot');
    } finally {
      setIsProcessing(false);
    }
  };

  const sendInitialText = async (transcript) => {
    try {
      const formData = new FormData();
      formData.append('transcript', transcript);
      formData.append('language', languageLabel);
      formData.append('is_ayush', isAyush);
      if (patient?.abhaId) formData.append('abha_id', patient.abhaId);
      if (patient?.name) formData.append('patient_name', patient.name);
      if (patient?.age) formData.append('age', patient.age);
      if (patient?.gender) formData.append('gender', patient.gender);
      if (patient?.phone) formData.append('phone', patient.phone);

      const response = await fetch(`${API_BASE_URL}/process-text`, { method: 'POST', body: formData });
      const data = await response.json();
      handleInitialResponse(data);
    } catch (error) {
      removeLastBotMessage();
      addMessage('Sorry, there was an error. Please try again.', 'bot');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleInitialResponse = (data) => {
    if (data.patient_id) setPatientId(data.patient_id);
    removeLastBotMessage();
    const q = data.next_question || getFirstFollowUp(languageLabel);
    addMessage(q, 'bot');
    setConversationPhase('follow-up');
    setConversationContext(`Initial complaint: ${data.extracted_complaint}\nBot: ${q}`);
    setProgress(30);
  };

  const sendFollowUpAudio = async (audioBlob) => {
    try {
      const formData = new FormData();
      formData.append('audio', audioBlob, 'recording.webm');
      formData.append('language', languageLabel);
      formData.append('patient_id', patientId);
      formData.append('conversation_context', conversationContext);
      formData.append('is_ayush', isAyush);
      formData.append('follow_up_count', followUpCount);
      const response = await fetch(`${API_BASE_URL}/follow-up`, { method: 'POST', body: formData });
      const data = await response.json();
      handleFollowUpResponse(data);
    } catch (error) {
      removeLastBotMessage();
      addMessage('Sorry, there was an error processing your response.', 'bot');
    } finally {
      setIsProcessing(false);
    }
  };

  const sendFollowUpText = async (transcript) => {
    try {
      const formData = new FormData();
      formData.append('transcript', transcript);
      formData.append('language', languageLabel);
      formData.append('patient_id', patientId);
      formData.append('conversation_context', conversationContext);
      formData.append('is_ayush', isAyush);
      formData.append('follow_up_count', followUpCount);
      const response = await fetch(`${API_BASE_URL}/follow-up-text`, { method: 'POST', body: formData });
      const data = await response.json();
      handleFollowUpResponse(data);
    } catch (error) {
      removeLastBotMessage();
      addMessage('Sorry, there was an error processing your response.', 'bot');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleFollowUpResponse = (data) => {
    removeLastBotMessage();
    if (!data) {
      addMessage('Sorry, there was an error processing your response.', 'bot');
      return;
    }
    setConversationContext(prev => prev + `\nPatient: ${data.transcript || ''}`);
    setFollowUpCount(prev => prev + 1);
    setProgress(prev => Math.min(prev + 20, 90));

    if (followUpCount >= 5 || (data.is_complete && followUpCount >= 5)) {
      addMessage('Thank you for providing all the information.', 'bot');
      setConversationPhase('complete');
      setProgress(100);
      setClinicalData(data);
      setTimeout(() => {
        navigate('/document-scan');
      }, 2000);
    } else {
      const nextQ = data.next_question || getNextQuestion(languageLabel, followUpCount);
      addMessage(nextQ, 'bot');
      setConversationContext(prev => prev + `\nBot: ${nextQ}`);
    }
  };

  // ── Quick Symptom Buttons ──
  const quickSymptoms = [
    { label: t('fever'), value: 'I have fever and chills' },
    { label: t('chest_pain'), value: 'I have chest pain' },
    { label: t('stomach'), value: 'I have stomach pain and nausea' },
    { label: t('cough'), value: 'I have continuous cough' },
    { label: t('joint_pain'), value: 'I have joint pain and stiffness' },
    { label: t('headache'), value: 'I have severe headache' },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', background: 'var(--color-bg)' }}>

      {/* Header */}
      <div style={{
        padding: 'var(--space-4) var(--space-6)',
        background: 'var(--color-surface)',
        borderBottom: '1px solid var(--color-border)',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between'
      }}>
        <div>
          <h3 style={{ fontSize: 'var(--font-size-lg)', fontWeight: 700, color: 'var(--color-text)' }}>
            🏥 {t('kiosk_title')}
          </h3>
          <span className="caption">
            {languageLabel} • {isAyush ? t('ayush_opd') : t('allo_opd')} •
            Phase: {conversationPhase.toUpperCase()}
          </span>
        </div>
        <div style={{ display: 'flex', gap: 'var(--space-2)', alignItems: 'center' }}>
          {patientId && <span className="badge badge-info">{patientId}</span>}
          <button className="btn btn-outline btn-sm" onClick={handleSkipDemo} disabled={isProcessing}>
            Skip (Demo)
          </button>
        </div>
      </div>

      {/* Progress Bar */}
      <div style={{ padding: '0 var(--space-6)', background: 'var(--color-surface)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-3)', padding: 'var(--space-2) 0 var(--space-3)' }}>
          <div className="progress-bar" style={{ flex: 1 }}>
            <div className="progress-bar-fill" style={{ width: `${progress}%` }} />
          </div>
          <span className="caption" style={{ fontWeight: 600, whiteSpace: 'nowrap' }}>
            {t('clinical_history')}: {progress}%
          </span>
        </div>
      </div>

      {/* Chat Messages */}
      <div className="chat-container" style={{ flex: 1, overflowY: 'auto', padding: 'var(--space-4) var(--space-6)' }}>
        {messages.map((msg, i) => (
          <div key={i} className={`chat-bubble chat-bubble-${msg.type}`}>
            {msg.type === 'bot' && (
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 'var(--space-1)' }}>
                <div style={{ fontSize: 'var(--font-size-xs)', fontWeight: 600, color: 'var(--color-primary)' }}>
                  🏥 MediKiosk AI
                </div>
                {msg.text !== 'Processing...' && msg.text !== 'Listening...' && (
                  <button
                    onClick={() => speakText(msg.text)}
                    style={{
                      background: 'none', border: 'none', cursor: 'pointer',
                      fontSize: 'var(--font-size-lg)', padding: '2px 6px',
                      borderRadius: 'var(--radius-sm)', opacity: 0.7,
                      transition: 'opacity 0.2s'
                    }}
                    onMouseEnter={e => e.target.style.opacity = 1}
                    onMouseLeave={e => e.target.style.opacity = 0.7}
                    title="Replay audio"
                  >
                    🔊
                  </button>
                )}
              </div>
            )}
            {msg.text === 'Processing...' ? (
              <div className="flex items-center gap-2">
                <div className="spinner" /> {t('processing')}
              </div>
            ) : (
              <div style={{ whiteSpace: 'pre-wrap' }}>{msg.text}</div>
            )}
          </div>
        ))}
        <div ref={chatEndRef} />
      </div>

      {/* Quick Symptoms (initial phase only) */}
      {conversationPhase === 'initial' && !isProcessing && (
        <div style={{ padding: 'var(--space-2) var(--space-6)', background: 'var(--color-surface)', borderTop: '1px solid var(--color-border)' }}>
          <p className="caption mb-2">{t('tap_symptom')}</p>
          <div style={{ display: 'flex', gap: 'var(--space-2)', flexWrap: 'wrap' }}>
            {quickSymptoms.map(s => (
              <button key={s.value} className="btn btn-sm btn-secondary"
                onClick={() => handleQuickAnswer(s.value)}
                style={{ fontSize: 'var(--font-size-sm)' }}>
                {s.label}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Quick Yes/No/Not Sure (follow-up phase) */}
      {conversationPhase === 'follow-up' && !isProcessing && (
        <div style={{ padding: 'var(--space-2) var(--space-6)', background: 'var(--color-surface)', borderTop: '1px solid var(--color-border)' }}>
          <div style={{ display: 'flex', gap: 'var(--space-2)' }}>
            <button className="btn btn-sm btn-success" onClick={() => handleQuickAnswer('Yes')} style={{ flex: 1 }}>
              {t('yes')}
            </button>
            <button className="btn btn-sm btn-danger" onClick={() => handleQuickAnswer('No')} style={{ flex: 1 }}>
              {t('no')}
            </button>
            <button className="btn btn-sm btn-secondary" onClick={() => handleQuickAnswer('Not sure')} style={{ flex: 1 }}>
              {t('not_sure')}
            </button>
          </div>
        </div>
      )}

      {/* Input Area */}
      {conversationPhase !== 'complete' && (
        <div style={{
          padding: 'var(--space-4) var(--space-6)',
          background: 'var(--color-surface)',
          borderTop: '1px solid var(--color-border)',
          display: 'flex', alignItems: 'center', gap: 'var(--space-3)'
        }}>
          <input
            className="input"
            placeholder={isRecording ? t('recording') : t('type_answer')}
            value={textInput}
            onChange={e => setTextInput(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleTextSubmit()}
            disabled={isProcessing || isRecording}
            style={{ flex: 1, minHeight: 48 }}
          />
          <button className="btn btn-primary btn-sm" onClick={handleTextSubmit}
            disabled={!textInput.trim() || isProcessing} style={{ minWidth: 48 }}>
            ➤
          </button>
          <button
            className={`mic-btn ${isRecording ? 'recording' : ''}`}
            onClick={handleMicClick}
            disabled={isProcessing}
            title={isRecording ? 'Stop recording' : 'Start recording'}
          >
            {isRecording ? '⏹' : '🎤'}
          </button>
        </div>
      )}
    </div>
  );
}

// ── Helpers ──
function getGreeting(lang) {
  const greetings = {
    'Hindi': 'नमस्ते! मैं MediKiosk AI हूँ। मैं आपकी चिकित्सा जानकारी एकत्र करने में मदद करूँगा।\n\nआपको आज डॉक्टर से मिलने की क्या समस्या है?',
    'Tamil': 'வணக்கம்! நான் MediKiosk AI. உங்கள் மருத்துவ வரலாற்றைச் சேகரிக்க உதவுவேன்.\n\nஇன்று நீங்கள் மருத்துவரை சந்திக்க என்ன பிரச்சனை?',
    'Telugu': 'నమస్కారం! నేను MediKiosk AI. మీ వైద్య చరిత్రను సేకరించడంలో సహాయపడతాను.\n\nమీరు ఈరోజు డాక్టర్‌ను ఎందుకు కలుస్తున్నారు?',
    'Bengali': 'নমস্কার! আমি MediKiosk AI। আপনার চিকিৎসা ইতিহাস সংগ্রহ করতে সাহায্য করব।\n\nআজ আপনি কি সমস্যায় ডাক্তারের কাছে এসেছেন?',
    'Marathi': 'नमस्कार! मी MediKiosk AI आहे। मी तुमचा वैद्यकीय इतिहास गोळा करण्यात मदत करीन।\n\nआज तुम्हाला डॉक्टरांना भेटायला काय त्रास आहे?',
  };
  return greetings[lang] || 'Hello! I am MediKiosk AI. I will help collect your medical history.\n\nWhat is the main health issue that brings you here today?';
}

function getFirstFollowUp(lang) {
  const followups = {
    'Hindi': 'अब मैं और जानना चाहता हूँ। क्या आप बता सकते हैं कि तकलीफ़ कहाँ है, कब शुरू हुई, और कितनी तेज़ है?',
    'Tamil': 'இப்போது நான் மேலும் புரிந்துகொள்ள விரும்புகிறேன். வலி எங்கு உள்ளது, எப்போது ஆரம்பித்தது, எவ்வளவு கடுமையானது?',
    'Telugu': 'ఇప్పుడు నేను మరింత అర్థం చేసుకోవాలనుకుంటున్నాను. నొప్పి ఎక్కడ ఉంది, ఎప్పుడు ప్రారంభమైంది, ఎంత తీవ్రమైనది?',
  };
  return followups[lang] || "Now I'd like to understand more. Can you describe where the discomfort is, when it started, and how severe it is?";
}

function getNextQuestion(lang, count) {
  const questions = {
    'Hindi': [
      'क्या आप बता सकते हैं कि यह दर्द या तकलीफ़ आपके शरीर में कहीं और फैलती है? क्या इसके साथ कोई और लक्षण हैं जैसे बुखार या उल्टी?',
      'क्या आपने इसके लिए कोई दवा ली है? क्या किसी चीज़ से आराम मिलता है या यह और बढ़ जाता है?',
      'क्या आपको पहले कभी ऐसी समस्या हुई है या आपके परिवार में किसी को ऐसी बीमारी है?',
      'क्या आपको किसी दवा, खाने-पीने की चीज़ या किसी अन्य चीज़ से कोई एलर्जी है?'
    ],
    'Tamil': [
      'இந்த அசௌகரியம் உங்கள் உடலின் வேறு எங்காவது பரவுகிறதா? காய்ச்சல் அல்லது வாந்தி போன்ற வேறு ஏதேனும் அறிகுறிகள் உள்ளதா?',
      'இதற்காக நீங்கள் ஏதேனும் மருந்து எடுத்துக்கொண்டீர்களா? எதையாவது செய்தால் இது குறைகிறதா அல்லது அதிகரிக்கிறதா?',
      'இதற்கு முன் உங்களுக்கு இந்த பிரச்சனை வந்திருக்கிறதா, அல்லது உங்கள் குடும்பத்தில் யாருக்காவது இதே போன்ற நிலை உள்ளதா?',
      'உங்களுக்கு ஏதேனும் மருந்து, உணவு அல்லது வேறு எதற்காவது ஒவ்வாமை (Allergy) உள்ளதா?'
    ],
    'Telugu': [
      'ఈ అసౌకర్యం మీ శరీరంలో మరెక్కడికైనా వ్యాపిస్తుందా? జ్వరం లేదా వాంతులు వంటి ఇతర లక్షణాలు ఏమైనా ఉన్నాయా?',
      'దీని కోసం మీరు ఏమైనా మందులు తీసుకున్నారా? ఏదైనా చేస్తే ఇది తగ్గుతుందా లేదా పెరుగుతుందా?',
      'మీకు ఇంతకు ముందు ఈ సమస్య ఎప్పుడైనా వచ్చిందా, లేదా మీ కుటుంబంలో ఎవరికైనా ఇలాంటి పరిస్థితి ఉందా?',
      'మీకు ఏదైనా మందు, ఆహారం లేదా మరేదైనా వస్తువు వల్ల అలెర్జీ ఉందా?'
    ]
  };

  const fallback = [
    "Can you describe if this discomfort spreads anywhere else in your body? Are there any other symptoms?",
    "Have you taken any medication for this? Does anything make it feel better or worse?",
    "Have you ever had this problem before, or does anyone in your family have a similar condition?",
    "Do you have any known allergies to medications, food, or anything else?"
  ];

  const langQs = questions[lang] || fallback;
  return langQs[count] || 'Thank you. Is there anything else you would like to add?';
}
