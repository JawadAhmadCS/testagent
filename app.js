const micButton = document.getElementById("micButton");
const statusEl = document.getElementById("status");
const userTextEl = document.getElementById("userText");
const botTextEl = document.getElementById("botText");
const languageSelectEl = document.getElementById("languageSelect");
const languageHintEl = document.getElementById("languageHint");
const latencySttModelEl = document.getElementById("latencySttModel");
const latencyLlmModelEl = document.getElementById("latencyLlmModel");
const latencyTtsModelEl = document.getElementById("latencyTtsModel");
const latencySttMsEl = document.getElementById("latencySttMs");
const latencyLlmMsEl = document.getElementById("latencyLlmMs");
const latencyTtsMsEl = document.getElementById("latencyTtsMs");
const latencyTotalMsEl = document.getElementById("latencyTotalMs");
const API_BASE = window.location.origin;

const ULTRA_FAST_MODE = true;
const MAX_HISTORY_MESSAGES = 8;
const MIN_RECORD_MS = 380;
const SILENCE_STOP_MS = 320;
const NO_SPEECH_STOP_MS = 2500;
const MAX_RECORD_MS = 7000;
const SILENCE_THRESHOLD = 0.015;
const BROWSER_TTS_RATE = 1.14;
const DEFAULT_LANGUAGE = "en";
const LANGUAGE_CONFIG = {
  en: {
    uiLabel: "English",
    sttCode: "en",
    speechSynthesisLang: "en-US",
    ttsModelLabel: "browser-speechSynthesis(en-US)",
  },
  he: {
    uiLabel: "Hebrew",
    sttCode: "he",
    speechSynthesisLang: "he-IL",
    ttsModelLabel: "browser-speechSynthesis(he-IL)",
  },
};

let mediaStream = null;
let mediaRecorder = null;
let chunks = [];
let isRecording = false;
let isProcessing = false;
let sessionActive = false;
let activeAudio = null;
let activeUtterance = null;
let sessionLanguage = DEFAULT_LANGUAGE;

let silenceIntervalId = null;
let maxRecordTimerId = null;
let recordingStartedAt = 0;
let lastVoiceAt = 0;
let speechDetected = false;

let audioContext = null;
let analyserNode = null;
let sourceNode = null;

const conversation = [];

function getLanguageConfig(languageKey) {
  return LANGUAGE_CONFIG[languageKey] || LANGUAGE_CONFIG[DEFAULT_LANGUAGE];
}

function getSelectedLanguageKey() {
  const key = languageSelectEl?.value || DEFAULT_LANGUAGE;
  return LANGUAGE_CONFIG[key] ? key : DEFAULT_LANGUAGE;
}

function setLanguageLock(locked) {
  if (languageSelectEl) {
    languageSelectEl.disabled = locked;
  }
  if (languageHintEl) {
    languageHintEl.textContent = locked
      ? `Language locked to ${getLanguageConfig(sessionLanguage).uiLabel} for this live session.`
      : "Language locks for current live session.";
  }
}

function formatMs(value) {
  if (typeof value !== "number" || Number.isNaN(value)) return "-";
  return `${Math.max(0, Math.round(value))} ms`;
}

function updateLatencyCard(data) {
  const timings = data?.timings || {};
  const models = data?.models || {};
  const languageCfg = getLanguageConfig(sessionLanguage);

  latencySttModelEl.textContent = models.transcribe || "-";
  latencyLlmModelEl.textContent = models.llm || "-";
  latencyTtsModelEl.textContent =
    models.tts || (ULTRA_FAST_MODE || data?.fastTts ? languageCfg.ttsModelLabel : "-");

  latencySttMsEl.textContent = formatMs(timings.transcribeMs);
  latencyLlmMsEl.textContent = formatMs(timings.llmMs);
  latencyTtsMsEl.textContent = formatMs(timings.ttsMs);
  latencyTotalMsEl.textContent = formatMs(timings.totalMs);
}

function resetForNewConversation() {
  conversation.length = 0;
  userTextEl.textContent = "-";
  botTextEl.textContent = "-";
  updateLatencyCard(null);
}

function updateStatus(text) {
  statusEl.textContent = text;
}

function appendHistory(role, content) {
  conversation.push({ role, content });
  while (conversation.length > MAX_HISTORY_MESSAGES) {
    conversation.shift();
  }
}

function getSupportedMimeType() {
  const types = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"];
  return types.find((type) => MediaRecorder.isTypeSupported(type)) || "";
}

async function reportClientError(payload) {
  try {
    await fetch(`${API_BASE}/api/client-error`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      keepalive: true,
    });
  } catch {
    // Ignore reporting failures to avoid recursive errors in client.
  }
}

async function parseJsonFromResponse(response) {
  const raw = await response.text();
  if (!raw) {
    throw new Error(`Empty response from server (${response.status}).`);
  }

  try {
    return JSON.parse(raw);
  } catch (error) {
    await reportClientError({
      source: "response-json-parse",
      status: response.status,
      statusText: response.statusText,
      bodyPreview: raw.slice(0, 1200),
      message: error.message,
    });
    throw new Error(`Server returned invalid JSON (${response.status}). Check terminal logs.`);
  }
}

window.addEventListener("error", (event) => {
  reportClientError({
    source: "window-error",
    message: event.message,
    file: event.filename,
    line: event.lineno,
    column: event.colno,
    stack: event.error?.stack || null,
  });
});

window.addEventListener("unhandledrejection", (event) => {
  const reason = event.reason;
  reportClientError({
    source: "unhandled-rejection",
    message: reason?.message || String(reason),
    stack: reason?.stack || null,
  });
});

async function ensureAudioReady() {
  if (!mediaStream) {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  }

  if (!audioContext) {
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    sourceNode = audioContext.createMediaStreamSource(mediaStream);
    analyserNode = audioContext.createAnalyser();
    analyserNode.fftSize = 2048;
    sourceNode.connect(analyserNode);
  }

  if (audioContext.state === "suspended") {
    await audioContext.resume();
  }

  if (!mediaRecorder) {
    const mimeType = getSupportedMimeType();
    mediaRecorder = mimeType
      ? new MediaRecorder(mediaStream, { mimeType })
      : new MediaRecorder(mediaStream);

    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) chunks.push(event.data);
    };

    mediaRecorder.onstop = async () => {
      clearSilenceWatch();
      let restartListening = false;

      try {
        if (!sessionActive) return;

        const audioType = mediaRecorder.mimeType || "audio/webm";
        const audioBlob = new Blob(chunks, { type: audioType });
        chunks = [];

        if (audioBlob.size < 1200) {
          updateStatus("I did not hear clear speech. Speak again...");
          await startListeningTurn();
          return;
        }

        isProcessing = true;
        updateStatus("Thinking...");

        const formData = new FormData();
        formData.append("audio", audioBlob, "speech.webm");
        formData.append("history", JSON.stringify(conversation));
        formData.append("fast", ULTRA_FAST_MODE ? "1" : "0");
        formData.append("language", getLanguageConfig(sessionLanguage).sttCode);

        const response = await fetch(`${API_BASE}/api/voice`, {
          method: "POST",
          body: formData,
        });

        const data = await parseJsonFromResponse(response);
        if (!response.ok) {
          throw new Error(data.error || "Request failed");
        }

        userTextEl.textContent = data.transcript || "-";
        botTextEl.textContent = data.reply || "-";

        appendHistory("user", data.transcript || "");
        appendHistory("assistant", data.reply || "");
        updateLatencyCard(data);

        const t = data.timings;
        if (t?.totalMs) {
          console.log("[latency-ms]", t);
        }

        await speakAssistantResponse(data);
        restartListening = sessionActive;
      } catch (error) {
        console.error(error);
        await reportClientError({
          source: "voice-onstop",
          message: error.message,
          stack: error.stack || null,
        });
        updateStatus(`Error: ${error.message}`);
      } finally {
        isProcessing = false;
        if (restartListening && sessionActive) {
          await startListeningTurn();
        }
      }
    };
  }
}

function detectSpeechRms() {
  const data = new Uint8Array(analyserNode.fftSize);
  analyserNode.getByteTimeDomainData(data);
  let sumSquares = 0;
  for (let i = 0; i < data.length; i += 1) {
    const centered = (data[i] - 128) / 128;
    sumSquares += centered * centered;
  }
  return Math.sqrt(sumSquares / data.length);
}

function clearSilenceWatch() {
  if (silenceIntervalId) {
    clearInterval(silenceIntervalId);
    silenceIntervalId = null;
  }
  if (maxRecordTimerId) {
    clearTimeout(maxRecordTimerId);
    maxRecordTimerId = null;
  }
}

function stopCurrentRecording() {
  if (!isRecording || !mediaRecorder) return;
  isRecording = false;
  clearSilenceWatch();
  if (mediaRecorder.state === "recording") {
    mediaRecorder.stop();
  }
}

function startSilenceWatch() {
  clearSilenceWatch();
  silenceIntervalId = setInterval(() => {
    if (!isRecording || !analyserNode) return;

    const now = Date.now();
    const rms = detectSpeechRms();
    const elapsed = now - recordingStartedAt;

    if (rms > SILENCE_THRESHOLD) {
      speechDetected = true;
      lastVoiceAt = now;
    }

    if (!speechDetected && elapsed >= NO_SPEECH_STOP_MS) {
      stopCurrentRecording();
      return;
    }

    if (speechDetected && elapsed >= MIN_RECORD_MS && now - lastVoiceAt >= SILENCE_STOP_MS) {
      stopCurrentRecording();
    }
  }, 80);

  maxRecordTimerId = setTimeout(() => {
    if (isRecording) {
      stopCurrentRecording();
    }
  }, MAX_RECORD_MS);
}

async function startListeningTurn() {
  if (!sessionActive || isProcessing || isRecording) return;

  await ensureAudioReady();

  chunks = [];
  speechDetected = false;
  recordingStartedAt = Date.now();
  lastVoiceAt = recordingStartedAt;
  isRecording = true;
  updateStatus("Listening...");

  mediaRecorder.start(100);
  startSilenceWatch();
}

function stopPlaybackIfAny() {
  if ("speechSynthesis" in window) {
    window.speechSynthesis.cancel();
  }
  activeUtterance = null;

  if (!activeAudio) return;
  activeAudio.pause();
  activeAudio.currentTime = 0;
  activeAudio = null;
}

async function speakWithBrowser(text, languageKey) {
  if (!text) return;
  if (!("speechSynthesis" in window)) return;

  stopPlaybackIfAny();
  updateStatus("Speaking...");
  const languageCfg = getLanguageConfig(languageKey);

  await new Promise((resolve) => {
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = BROWSER_TTS_RATE;
    utterance.pitch = 1;
    utterance.lang = languageCfg.speechSynthesisLang;
    utterance.onend = resolve;
    utterance.onerror = resolve;
    activeUtterance = utterance;
    window.speechSynthesis.speak(utterance);
  });

  activeUtterance = null;
}

async function playAssistantAudio(audioBase64, audioMime) {
  if (!audioBase64) return;

  stopPlaybackIfAny();
  updateStatus("Speaking...");

  const audio = new Audio(`data:${audioMime};base64,${audioBase64}`);
  activeAudio = audio;

  try {
    await audio.play();
    await new Promise((resolve) => {
      audio.onended = resolve;
      audio.onerror = resolve;
    });
  } finally {
    activeAudio = null;
  }
}

async function speakAssistantResponse(data) {
  const preferBrowser = ULTRA_FAST_MODE || data.fastTts || !data.audioBase64;
  if (preferBrowser) {
    await speakWithBrowser(data.reply || "", sessionLanguage);
    return;
  }
  await playAssistantAudio(data.audioBase64, data.audioMime || "audio/mp3");
}

async function startSession() {
  if (sessionActive) return;
  try {
    await fetch(`${API_BASE}/api/health`, { cache: "no-store" });
  } catch {
    // Ignore warm-up failures; main loop will surface errors.
  }

  sessionLanguage = getSelectedLanguageKey();
  sessionActive = true;
  setLanguageLock(true);
  micButton.classList.add("recording");
  updateStatus(`Live mode ON (${getLanguageConfig(sessionLanguage).uiLabel}). Speak naturally...`);
  await startListeningTurn();
}

function stopSession() {
  sessionActive = false;
  isProcessing = false;
  stopCurrentRecording();
  stopPlaybackIfAny();
  micButton.classList.remove("recording");
  setLanguageLock(false);
  updateStatus("Live mode OFF. Tap mic to start");
}

async function toggleSession() {
  try {
    if (!sessionActive) {
      await startSession();
      return;
    }
    stopSession();
  } catch (error) {
    console.error(error);
    await reportClientError({
      source: "toggle-session",
      message: error.message,
      stack: error.stack || null,
    });
    updateStatus(`Mic error: ${error.message}`);
    stopSession();
  }
}

if (languageSelectEl) {
  languageSelectEl.value = getSelectedLanguageKey();
  languageSelectEl.addEventListener("change", () => {
    if (sessionActive) {
      languageSelectEl.value = sessionLanguage;
      return;
    }
    sessionLanguage = getSelectedLanguageKey();
    resetForNewConversation();
    updateStatus(`Language set to ${getLanguageConfig(sessionLanguage).uiLabel}. Tap mic to start.`);
  });
}

sessionLanguage = getSelectedLanguageKey();
setLanguageLock(false);
updateLatencyCard(null);

micButton.addEventListener("click", toggleSession);
