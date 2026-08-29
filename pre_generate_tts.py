import os
import hashlib
from gtts import gTTS

TTS_CACHE_DIR = os.path.join("uploads", "tts_cache")
os.makedirs(TTS_CACHE_DIR, exist_ok=True)

phrases = [
    # Hindi phrases
    ("नमस्ते! मैं MediKiosk AI हूँ। मैं आपकी चिकित्सा जानकारी एकत्र करने में मदद करूँगा।\n\nआपको आज डॉक्टर से मिलने की क्या समस्या है?", "hi"),
    ("नमस्ते! मैं MediKiosk AI हूँ। मैं आपकी चिकित्सा जानकारी एकत्र करने में मदद करूँगा। आपको आज डॉक्टर से मिलने की क्या समस्या है?", "hi"),
    ("यह तकलीफ़ आपको कब से हो रही है?", "hi"),
    ("क्या यह दर्द शरीर के किसी और हिस्से में भी जाता है?", "hi"),
    ("इसके साथ और कोई तकलीफ़ है? जैसे बुखार, उल्टी, या कमज़ोरी?", "hi"),
    ("क्या आपने इसके लिए कोई दवा ली है? किसी चीज़ से आराम मिलता है या तकलीफ़ बढ़ती है?", "hi"),
    ("क्या आपको पहले कोई बीमारी रही है? परिवार में किसी को कोई बीमारी है? क्या आप धूम्रपान या शराब का सेवन करते हैं?", "hi"),
    ("क्या आपको किसी दवा या खाने की चीज़ से एलर्जी है?", "hi"),
    ("और कुछ बताना चाहेंगे?", "hi"),
    ("Thank you. Let me ask a few follow-up questions.", "en"),
    ("Thank you for providing all the information.", "en"),
]

print("Generating TTS cache for Kiosk phrases...")
for idx, (text, lang) in enumerate(phrases):
    cache_key = hashlib.md5(f"{lang}_{text.strip()}".encode("utf-8")).hexdigest()
    out_file = os.path.join(TTS_CACHE_DIR, f"{cache_key}.mp3")
    if not os.path.exists(out_file):
        try:
            print(f"Generating phrase #{idx} [{lang}]...")
            tts = gTTS(text=text.strip(), lang=lang)
            tts.save(out_file)
            print(f"Saved: {cache_key}.mp3")
        except Exception as e:
            print(f"Error generating phrase #{idx}: {e}")
    else:
        print(f"Already exists: {cache_key}.mp3")

print("Pre-generation complete!")
