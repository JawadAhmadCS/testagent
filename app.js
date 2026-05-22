const micButton = document.getElementById("micButton");
const statusEl = document.getElementById("status");
const userTextEl = document.getElementById("userText");
const botTextEl = document.getElementById("botText");
const languageSelectEl = document.getElementById("languageSelect");
const languageHintEl = document.getElementById("languageHint");
const voiceLabelEl = document.getElementById("voiceLabel");
const voiceSelectEl = document.getElementById("voiceSelect");
const voiceHintEl = document.getElementById("voiceHint");
const ttsEngineSelectEl = document.getElementById("ttsEngineSelect");
const voiceSpeedRangeEl = document.getElementById("voiceSpeedRange");
const voiceSpeedValueEl = document.getElementById("voiceSpeedValue");
const voicePitchRangeEl = document.getElementById("voicePitchRange");
const voicePitchValueEl = document.getElementById("voicePitchValue");
const voiceVolumeRangeEl = document.getElementById("voiceVolumeRange");
const voiceVolumeValueEl = document.getElementById("voiceVolumeValue");
const voiceControlsHintEl = document.getElementById("voiceControlsHint");
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
const MIN_RECORD_MS = 220;
const SILENCE_STOP_MS = 220;
const NO_SPEECH_STOP_MS = 900;
const MAX_RECORD_MS = 4500;
const SILENCE_THRESHOLD = 0.015;
const BARGE_IN_TRIGGER_MS = 220;
const BROWSER_TTS_RATE = 1.14;
const MEDIA_RECORDER_AUDIO_BPS = 24000;
const VOICE_SETTINGS_STORAGE_KEY = "voice.settings.v1";
const DEFAULT_VOICE_SETTINGS = Object.freeze({
  engine: "auto",
  rate: 1,
  pitch: 1,
  volume: 1,
});
const DEFAULT_LANGUAGE = "he";
const DEFAULT_ENGLISH_VOICE = "en-US-Chirp3-HD-Kore";
const DEFAULT_HEBREW_VOICES = [
  "he-IL-Wavenet-C",
  "he-IL-Wavenet-A",
  "he-IL-Wavenet-B",
  "he-IL-Wavenet-D",
];
const DEFAULT_HEBREW_VOICE = DEFAULT_HEBREW_VOICES[0];
const CHIRP3_HD_VOICE_GROUPS_EN = {
  female: [
    "Achernar",
    "Aoede",
    "Autonoe",
    "Callirrhoe",
    "Despina",
    "Erinome",
    "Gacrux",
    "Kore",
    "Laomedeia",
    "Leda",
    "Pulcherrima",
    "Sulafat",
    "Vindemiatrix",
    "Zephyr",
  ],
  male: [
    "Achird",
    "Algenib",
    "Algieba",
    "Alnilam",
    "Charon",
    "Enceladus",
    "Fenrir",
    "Iapetus",
    "Orus",
    "Puck",
    "Rasalgethi",
    "Sadachbia",
    "Sadaltager",
    "Schedar",
    "Umbriel",
    "Zubenelgenubi",
  ],
};
const CHIRP3_HD_POPULAR_EN = new Set([
  "Aoede",
  "Kore",
  "Leda",
  "Zephyr",
  "Charon",
  "Fenrir",
  "Orus",
  "Puck",
]);
const SERVER_TTS_LANGUAGES = new Set(["he"]);
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
let activeVoiceRequestController = null;
let pendingPlaybackResolve = null;
let sessionLanguage = DEFAULT_LANGUAGE;
let voiceCatalog = null;
const selectedVoiceByLanguage = {};

let silenceIntervalId = null;
let maxRecordTimerId = null;
let bargeInIntervalId = null;
let bargeInVoiceStartedAt = 0;
let isHandlingBargeIn = false;
let recordingStartedAt = 0;
let lastVoiceAt = 0;
let speechDetected = false;

let audioContext = null;
let analyserNode = null;
let sourceNode = null;

const conversation = [];
let voiceSettings = loadVoiceSettings();

function getLanguageConfig(languageKey) {
  return LANGUAGE_CONFIG[languageKey] || LANGUAGE_CONFIG[DEFAULT_LANGUAGE];
}

function clampNumber(value, min, max, fallback) {
  const num = Number(value);
  if (!Number.isFinite(num)) return fallback;
  return Math.min(max, Math.max(min, num));
}

function formatVoiceMultiplier(value) {
  return `${value.toFixed(2)}x`;
}

function formatVoiceVolume(value) {
  return `${Math.round(value * 100)}%`;
}

function loadVoiceSettings() {
  const fallback = { ...DEFAULT_VOICE_SETTINGS };

  try {
    const raw = localStorage.getItem(VOICE_SETTINGS_STORAGE_KEY);
    if (!raw) return fallback;
    const parsed = JSON.parse(raw);
    const engine = ["auto", "browser", "server"].includes(parsed?.engine)
      ? parsed.engine
      : fallback.engine;

    return {
      engine,
      rate: clampNumber(parsed?.rate, 0.5, 2, fallback.rate),
      pitch: clampNumber(parsed?.pitch, 0.5, 2, fallback.pitch),
      volume: clampNumber(parsed?.volume, 0, 1, fallback.volume),
    };
  } catch {
    return fallback;
  }
}

function saveVoiceSettings() {
  try {
    localStorage.setItem(VOICE_SETTINGS_STORAGE_KEY, JSON.stringify(voiceSettings));
  } catch {
    // Ignore storage errors (private mode / quota) and continue with runtime values.
  }
}

function syncVoiceControlsFromState() {
  if (ttsEngineSelectEl) {
    ttsEngineSelectEl.value = voiceSettings.engine;
  }
  if (voiceSpeedRangeEl) {
    voiceSpeedRangeEl.value = String(voiceSettings.rate);
  }
  if (voiceSpeedValueEl) {
    voiceSpeedValueEl.textContent = formatVoiceMultiplier(voiceSettings.rate);
  }
  if (voicePitchRangeEl) {
    voicePitchRangeEl.value = String(voiceSettings.pitch);
  }
  if (voicePitchValueEl) {
    voicePitchValueEl.textContent = formatVoiceMultiplier(voiceSettings.pitch);
  }
  if (voiceVolumeRangeEl) {
    voiceVolumeRangeEl.value = String(voiceSettings.volume);
  }
  if (voiceVolumeValueEl) {
    voiceVolumeValueEl.textContent = formatVoiceVolume(voiceSettings.volume);
  }
}

function updateVoiceControlsHint(languageKey) {
  if (!voiceControlsHintEl) return;

  if (voiceSettings.engine === "browser") {
    voiceControlsHintEl.textContent =
      "Browser mode: speed, pitch, and volume apply directly in this browser.";
    return;
  }

  if (voiceSettings.engine === "server") {
    voiceControlsHintEl.textContent =
      "Server mode: Google TTS applies speed, pitch, and volume for the selected voice.";
    return;
  }

  if (languageKey === "he") {
    voiceControlsHintEl.textContent =
      "Auto mode: Hebrew prefers server voice for better reliability.";
    return;
  }

  if (getSelectedVoice(languageKey)) {
    voiceControlsHintEl.textContent =
      "Auto mode: selected voice uses server TTS so male/female voice choice is preserved.";
    return;
  }

  voiceControlsHintEl.textContent =
    "Auto mode: uses best available engine and applies your settings on next reply.";
}

function updateVoiceControlAvailability(languageKey) {
  if (!ttsEngineSelectEl) return;

  const serverOption = Array.from(ttsEngineSelectEl.options).find(
    (option) => option.value === "server"
  );
  const serverCapable = shouldPreferServerTts(languageKey);

  if (serverOption) {
    serverOption.disabled = !serverCapable;
  }

  if (!serverCapable && voiceSettings.engine === "server") {
    voiceSettings.engine = "auto";
    saveVoiceSettings();
    syncVoiceControlsFromState();
  }

  updateVoiceControlsHint(languageKey);
}

function applyVoiceControlChange() {
  if (ttsEngineSelectEl) {
    voiceSettings.engine = ttsEngineSelectEl.value;
  }
  if (voiceSpeedRangeEl) {
    voiceSettings.rate = clampNumber(voiceSpeedRangeEl.value, 0.5, 2, DEFAULT_VOICE_SETTINGS.rate);
  }
  if (voicePitchRangeEl) {
    voiceSettings.pitch = clampNumber(
      voicePitchRangeEl.value,
      0.5,
      2,
      DEFAULT_VOICE_SETTINGS.pitch
    );
  }
  if (voiceVolumeRangeEl) {
    voiceSettings.volume = clampNumber(
      voiceVolumeRangeEl.value,
      0,
      1,
      DEFAULT_VOICE_SETTINGS.volume
    );
  }

  saveVoiceSettings();
  syncVoiceControlsFromState();
  updateVoiceControlAvailability(sessionLanguage || getSelectedLanguageKey());
}

function getSelectedLanguageKey() {
  const key = languageSelectEl?.value || DEFAULT_LANGUAGE;
  return LANGUAGE_CONFIG[key] ? key : DEFAULT_LANGUAGE;
}

function buildFallbackVoiceCatalog() {
  const voicesEn = Object.entries(CHIRP3_HD_VOICE_GROUPS_EN).flatMap(([gender, names]) =>
    names.map((name) => ({
      id: `en-US-Chirp3-HD-${name}`,
      name,
      gender,
      popular: CHIRP3_HD_POPULAR_EN.has(name),
    }))
  );

  return {
    en: {
      key: "en",
      label: "English",
      defaultVoice: DEFAULT_ENGLISH_VOICE,
      voices: voicesEn,
    },
    he: {
      key: "he",
      label: "Hebrew",
      defaultVoice: DEFAULT_HEBREW_VOICE,
      voices: DEFAULT_HEBREW_VOICES.map((id) => ({
        id,
        name: id,
        gender: "default",
        popular: id === DEFAULT_HEBREW_VOICE,
      })),
    },
  };
}

function getVoiceConfig(languageKey) {
  if (!voiceCatalog) {
    voiceCatalog = buildFallbackVoiceCatalog();
  }
  return voiceCatalog[languageKey] || { voices: [], defaultVoice: "" };
}

function setVoiceHint(languageKey) {
  if (!voiceHintEl) return;
  if (voiceLabelEl) {
    voiceLabelEl.textContent =
      languageKey === "en" ? "Google Chirp 3 HD Voice (English)" : "Google Chirp Voice (Hebrew)";
  }
  if (voiceSelectEl) {
    voiceSelectEl.setAttribute(
      "aria-label",
      languageKey === "en" ? "Select Google Chirp voice" : "Configured Google voice for Hebrew"
    );
  }

  if (languageKey === "en") {
    const selectedVoice = selectedVoiceByLanguage.en;
    if (selectedVoice) {
      const shortName = selectedVoice.replace("en-US-Chirp3-HD-", "");
      voiceHintEl.textContent = `Voice: ${shortName}. Change anytime to compare tone on the next reply.`;
      return;
    }
    voiceHintEl.textContent = "Select a Google Chirp 3 HD voice for tone testing.";
    return;
  }

  const selectedVoice = selectedVoiceByLanguage[languageKey];
  if (selectedVoice) {
    voiceHintEl.textContent = `Voice: ${selectedVoice}. This voice is used for Hebrew replies.`;
    return;
  }
  voiceHintEl.textContent = "Select a Hebrew Google voice for replies.";
}

function getSelectedVoice(languageKey) {
  if (languageKey === "en") {
    return selectedVoiceByLanguage.en || "";
  }
  return selectedVoiceByLanguage[languageKey] || "";
}

function shouldPreferServerTts(languageKey) {
  if (SERVER_TTS_LANGUAGES.has(languageKey)) return true;
  return Boolean(getSelectedVoice(languageKey));
}

function resolveUseServerTtsForRequest(languageKey, shouldUseFast) {
  if (voiceSettings.engine === "browser") return false;
  if (voiceSettings.engine === "server") return true;

  const hasSelectedVoice = Boolean(getSelectedVoice(languageKey));

  // In auto mode, honor explicit voice choices (especially EN male/female Chirp voices).
  if (hasSelectedVoice) return true;

  // Hebrew remains server-first for reliability.
  if (languageKey === "he") return true;

  // For other cases keep ultra-fast browser preference unless fast mode is off.
  return shouldPreferServerTts(languageKey) && !shouldUseFast;
}

function createVoiceOptionElement(voice) {
  const option = document.createElement("option");
  option.value = voice.id;
  const popularTag = voice.popular ? " (popular)" : "";
  option.textContent = `${voice.name}${popularTag}`;
  return option;
}

function populateVoiceSelect(languageKey) {
  if (!voiceSelectEl) return;

  const voiceCfg = getVoiceConfig(languageKey);
  const voices = Array.isArray(voiceCfg.voices) ? voiceCfg.voices : [];

  voiceSelectEl.innerHTML = "";

  if (voices.length === 0) {
    voiceSelectEl.disabled = true;
    const noneOption = document.createElement("option");
    noneOption.value = "";
    noneOption.textContent = "No Chirp list for this language";
    voiceSelectEl.appendChild(noneOption);
    setVoiceHint(languageKey);
    updateVoiceControlAvailability(languageKey);
    return;
  }

  voiceSelectEl.disabled = false;

  let preferredVoice = selectedVoiceByLanguage[languageKey] || voiceCfg.defaultVoice || "";
  const voicesById = new Map(voices.map((voice) => [voice.id, voice]));
  if (!voicesById.has(preferredVoice)) {
    preferredVoice = voicesById.has(DEFAULT_ENGLISH_VOICE)
      ? DEFAULT_ENGLISH_VOICE
      : voices[0]?.id || "";
  }
  selectedVoiceByLanguage[languageKey] = preferredVoice;

  const hasGenderGroups = voices.some((voice) => voice.gender === "female" || voice.gender === "male");
  if (hasGenderGroups) {
    const femaleVoices = voices.filter((voice) => voice.gender === "female");
    const maleVoices = voices.filter((voice) => voice.gender === "male");

    const groups = [
      { label: "Female", list: femaleVoices },
      { label: "Male", list: maleVoices },
    ];

    groups.forEach(({ label, list }) => {
      if (list.length === 0) return;
      const group = document.createElement("optgroup");
      group.label = label;
      list.forEach((voice) => {
        group.appendChild(createVoiceOptionElement(voice));
      });
      voiceSelectEl.appendChild(group);
    });
  } else {
    voices.forEach((voice) => {
      voiceSelectEl.appendChild(createVoiceOptionElement(voice));
    });
  }

  voiceSelectEl.value = preferredVoice;
  setVoiceHint(languageKey);
  updateVoiceControlAvailability(languageKey);
}

async function loadVoiceCatalog() {
  try {
    const response = await fetch(`${API_BASE}/api/voices`, { cache: "no-store" });
    const data = await parseJsonFromResponse(response);
    if (response.ok && data?.languages) {
      voiceCatalog = data.languages;
      return;
    }
  } catch (error) {
    await reportClientError({
      source: "load-voice-catalog",
      message: error.message,
      stack: error.stack || null,
    });
  }

  voiceCatalog = buildFallbackVoiceCatalog();
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

function interruptCurrentTurn() {
  if (activeVoiceRequestController) {
    activeVoiceRequestController.abort();
    activeVoiceRequestController = null;
  }
  stopPlaybackIfAny();
}

function stopBargeInWatch() {
  if (bargeInIntervalId) {
    clearInterval(bargeInIntervalId);
    bargeInIntervalId = null;
  }
  bargeInVoiceStartedAt = 0;
}

async function handleBargeInDetected() {
  if (!sessionActive || !isProcessing || isRecording || isHandlingBargeIn) return;

  isHandlingBargeIn = true;
  try {
    updateStatus("Heard you. Interrupting current response...");
    interruptCurrentTurn();
    await startListeningTurn({ allowDuringProcessing: true });
  } finally {
    isHandlingBargeIn = false;
  }
}

function startBargeInWatch() {
  if (bargeInIntervalId) return;

  bargeInIntervalId = setInterval(() => {
    if (!sessionActive || !isProcessing || isRecording || !analyserNode) {
      bargeInVoiceStartedAt = 0;
      return;
    }

    const rms = detectSpeechRms();
    const now = Date.now();

    if (rms > SILENCE_THRESHOLD) {
      if (!bargeInVoiceStartedAt) {
        bargeInVoiceStartedAt = now;
        return;
      }

      if (now - bargeInVoiceStartedAt >= BARGE_IN_TRIGGER_MS) {
        bargeInVoiceStartedAt = 0;
        handleBargeInDetected();
      }
      return;
    }

    bargeInVoiceStartedAt = 0;
  }, 80);
}

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
    const recorderOptions = mimeType
      ? { mimeType, audioBitsPerSecond: MEDIA_RECORDER_AUDIO_BPS }
      : { audioBitsPerSecond: MEDIA_RECORDER_AUDIO_BPS };
    mediaRecorder = new MediaRecorder(mediaStream, recorderOptions);

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
          await startListeningTurn({ allowDuringProcessing: true });
          return;
        }

        isProcessing = true;
        updateStatus("Thinking...");

        const formData = new FormData();
        formData.append("audio", audioBlob, "speech.webm");
        formData.append("history", JSON.stringify(conversation));
        const shouldUseFast = ULTRA_FAST_MODE;
        const shouldUseServerTts = resolveUseServerTtsForRequest(sessionLanguage, shouldUseFast);
        formData.append("fast", shouldUseFast ? "1" : "0");
        formData.append("useServerTts", shouldUseServerTts ? "1" : "0");
        formData.append("ttsEngine", voiceSettings.engine);
        formData.append("ttsRate", String(voiceSettings.rate));
        formData.append("ttsPitch", String(voiceSettings.pitch));
        formData.append("ttsVolume", String(voiceSettings.volume));
        formData.append("language", getLanguageConfig(sessionLanguage).sttCode);
        const selectedVoice = getSelectedVoice(sessionLanguage);
        if (selectedVoice) {
          formData.append("voice", selectedVoice);
        }

        const requestController = new AbortController();
        activeVoiceRequestController = requestController;

        let response;
        try {
          response = await fetch(`${API_BASE}/api/voice`, {
            method: "POST",
            body: formData,
            signal: requestController.signal,
          });
        } finally {
          if (activeVoiceRequestController === requestController) {
            activeVoiceRequestController = null;
          }
        }

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
        if (error?.name === "AbortError") {
          return;
        }

        console.error(error);
        await reportClientError({
          source: "voice-onstop",
          message: error.message,
          stack: error.stack || null,
        });
        updateStatus(`Error: ${error.message}`);
      } finally {
        isProcessing = false;
        if (restartListening && sessionActive && !isRecording) {
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

async function startListeningTurn(options = {}) {
  const allowDuringProcessing = Boolean(options.allowDuringProcessing);
  if (!sessionActive || isRecording) return;
  if (isProcessing && !allowDuringProcessing) return;

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
  if (pendingPlaybackResolve) {
    const resolve = pendingPlaybackResolve;
    pendingPlaybackResolve = null;
    resolve();
  }

  if ("speechSynthesis" in window) {
    window.speechSynthesis.cancel();
  }
  activeUtterance = null;

  if (!activeAudio) return;
  activeAudio.pause();
  activeAudio.currentTime = 0;
  activeAudio = null;
}

async function speakWithBrowser(text, languageKey, settings = DEFAULT_VOICE_SETTINGS) {
  if (!text) return;
  if (!("speechSynthesis" in window)) return;

  stopPlaybackIfAny();
  updateStatus("Speaking...");
  const languageCfg = getLanguageConfig(languageKey);

  await new Promise((resolve) => {
    pendingPlaybackResolve = resolve;
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = clampNumber(settings.rate, 0.1, 10, BROWSER_TTS_RATE);
    utterance.pitch = clampNumber(settings.pitch, 0, 2, 1);
    utterance.volume = clampNumber(settings.volume, 0, 1, 1);
    utterance.lang = languageCfg.speechSynthesisLang;
    utterance.onend = () => {
      if (pendingPlaybackResolve === resolve) {
        pendingPlaybackResolve = null;
      }
      resolve();
    };
    utterance.onerror = () => {
      if (pendingPlaybackResolve === resolve) {
        pendingPlaybackResolve = null;
      }
      resolve();
    };
    activeUtterance = utterance;
    window.speechSynthesis.speak(utterance);
  });

  activeUtterance = null;
}

async function playAssistantAudio(audioBase64, audioMime, settings = DEFAULT_VOICE_SETTINGS) {
  if (!audioBase64) return;

  stopPlaybackIfAny();
  updateStatus("Speaking...");

  const audio = new Audio(`data:${audioMime};base64,${audioBase64}`);
  audio.volume = clampNumber(settings.volume, 0, 1, 1);
  activeAudio = audio;

  try {
    await audio.play();
    await new Promise((resolve) => {
      pendingPlaybackResolve = resolve;
      audio.onended = () => {
        if (pendingPlaybackResolve === resolve) {
          pendingPlaybackResolve = null;
        }
        resolve();
      };
      audio.onerror = () => {
        if (pendingPlaybackResolve === resolve) {
          pendingPlaybackResolve = null;
        }
        resolve();
      };
      audio.onpause = () => {
        if (pendingPlaybackResolve === resolve) {
          pendingPlaybackResolve = null;
        }
        resolve();
      };
    });
  } finally {
    activeAudio = null;
    pendingPlaybackResolve = null;
  }
}

async function speakAssistantResponse(data) {
  const requestedBrowser = voiceSettings.engine === "browser";
  const requestedServer = voiceSettings.engine === "server";
  const preferBrowser =
    requestedBrowser ||
    (!requestedServer && !shouldPreferServerTts(sessionLanguage) && ULTRA_FAST_MODE) ||
    data.fastTts ||
    !data.audioBase64;
  if (preferBrowser) {
    await speakWithBrowser(data.reply || "", sessionLanguage, voiceSettings);
    return;
  }
  await playAssistantAudio(data.audioBase64, data.audioMime || "audio/mp3", voiceSettings);
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
  await ensureAudioReady();
  startBargeInWatch();
  setLanguageLock(true);
  micButton.classList.add("recording");
  updateStatus(`Live mode ON (${getLanguageConfig(sessionLanguage).uiLabel}). Speak naturally...`);
  await startListeningTurn();
}

function stopSession() {
  sessionActive = false;
  isProcessing = false;
  interruptCurrentTurn();
  stopCurrentRecording();
  stopBargeInWatch();
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
    populateVoiceSelect(sessionLanguage);
    resetForNewConversation();
    updateStatus(`Language set to ${getLanguageConfig(sessionLanguage).uiLabel}. Tap mic to start.`);
  });
}

if (voiceSelectEl) {
  voiceSelectEl.addEventListener("change", () => {
    const languageKey = sessionLanguage || getSelectedLanguageKey();
    selectedVoiceByLanguage[languageKey] = voiceSelectEl.value;
    setVoiceHint(languageKey);
    updateVoiceControlAvailability(languageKey);
    updateStatus("Voice updated. Next response will use the selected voice.");
  });
}

async function initializeVoiceSelector() {
  await loadVoiceCatalog();
  populateVoiceSelect(sessionLanguage);
}

function initializeVoiceControls() {
  syncVoiceControlsFromState();
  updateVoiceControlAvailability(sessionLanguage || getSelectedLanguageKey());

  if (ttsEngineSelectEl) {
    ttsEngineSelectEl.addEventListener("change", () => {
      applyVoiceControlChange();
      updateStatus("Voice engine updated. Next response will use it.");
    });
  }

  if (voiceSpeedRangeEl) {
    voiceSpeedRangeEl.addEventListener("input", () => {
      applyVoiceControlChange();
    });
  }

  if (voicePitchRangeEl) {
    voicePitchRangeEl.addEventListener("input", () => {
      applyVoiceControlChange();
    });
  }

  if (voiceVolumeRangeEl) {
    voiceVolumeRangeEl.addEventListener("input", () => {
      applyVoiceControlChange();
    });
  }
}

sessionLanguage = getSelectedLanguageKey();
setLanguageLock(false);
updateLatencyCard(null);
initializeVoiceSelector();
initializeVoiceControls();

micButton.addEventListener("click", toggleSession);
