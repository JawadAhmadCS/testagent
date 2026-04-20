import "dotenv/config";
import express from "express";
import multer from "multer";
import { GoogleAuth } from "google-auth-library";
import { performance } from "node:perf_hooks";
import { join } from "node:path";

const app = express();
const upload = multer({ limits: { fileSize: 2 * 1024 * 1024 } });
const port = process.env.PORT || 3000;

app.use(express.static("."));
app.use(express.json());
app.use((req, res, next) => {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET,POST,OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");

  if (req.method === "OPTIONS") {
    return res.status(204).end();
  }
  return next();
});

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const GOOGLE_SERVICE_ACCOUNT_JSON = process.env.GOOGLE_SERVICE_ACCOUNT_JSON;
const GOOGLE_SERVICE_ACCOUNT_B64 = process.env.GOOGLE_SERVICE_ACCOUNT_B64;

function resolveGoogleCredentialsFromEnv() {
  const rawFromEnv = GOOGLE_SERVICE_ACCOUNT_JSON?.trim();
  const b64FromEnv = GOOGLE_SERVICE_ACCOUNT_B64?.trim();

  if (!rawFromEnv && !b64FromEnv) return null;

  try {
    const jsonText = rawFromEnv || Buffer.from(b64FromEnv, "base64").toString("utf8");
    const parsed = JSON.parse(jsonText);
    if (typeof parsed.private_key === "string") {
      parsed.private_key = parsed.private_key.replace(/\\n/g, "\n");
    }
    return parsed;
  } catch (error) {
    console.error("Failed to parse Google service account from env:", error.message || error);
    return null;
  }
}

const GOOGLE_CREDENTIALS = resolveGoogleCredentialsFromEnv();
const GOOGLE_AUTH_SOURCE = GOOGLE_CREDENTIALS ? "env-inline" : "google-application-credentials";
const GOOGLE_CLOUD_PROJECT_ID = process.env.GOOGLE_CLOUD_PROJECT_ID || GOOGLE_CREDENTIALS?.project_id;
const CHIRP_VOICE_EN = process.env.CHIRP_VOICE_EN || process.env.CHIRP_VOICE || "en-US-Chirp3-HD-Kore";
const CHIRP_VOICE_HE = process.env.CHIRP_VOICE_HE || "";
const GOOGLE_TTS_ENDPOINT =
  process.env.GOOGLE_TTS_ENDPOINT || "https://texttospeech.googleapis.com/v1/text:synthesize";
const TRANSCRIBE_MODEL = process.env.TRANSCRIBE_MODEL || "gpt-4o-transcribe";
const DEFAULT_LLM_MODEL = process.env.DEFAULT_LLM_MODEL || "gpt-4o";
const FAST_LLM_MODEL = process.env.FAST_LLM_MODEL || "gpt-4o-mini";
const FAST_RESPONSE_MODE = process.env.FAST_RESPONSE_MODE !== "0";
const FAST_MAX_TOKENS = Number(process.env.FAST_MAX_TOKENS || 48);
const DEFAULT_LANGUAGE = "en";
const HEBREW_REGEX = /[\u0590-\u05FF]/g;
const LATIN_REGEX = /[A-Za-z]/g;
const LANGUAGE_CONFIG = {
  en: {
    key: "en",
    label: "English",
    sttCode: "en",
    speechSynthesisLang: "en-US",
    chirpLanguageCode: "en-US",
    chirpVoice: CHIRP_VOICE_EN,
  },
  he: {
    key: "he",
    label: "Hebrew",
    sttCode: "he",
    speechSynthesisLang: "he-IL",
    chirpLanguageCode: "he-IL",
    chirpVoice: CHIRP_VOICE_HE,
  },
};

function logError(label, error, context = null) {
  console.error(`\n[${new Date().toISOString()}] ${label}`);
  if (context) {
    console.error("Context:", context);
  }
  if (error instanceof Error) {
    console.error(error.stack || error.message);
  } else {
    console.error(error);
  }
}

function resolveConversationLanguage(inputLanguage) {
  const value = typeof inputLanguage === "string" ? inputLanguage.trim().toLowerCase() : "";
  if (value && LANGUAGE_CONFIG[value]) {
    return value;
  }
  return DEFAULT_LANGUAGE;
}

function countMatches(text, regex) {
  return (text.match(regex) || []).length;
}

function isTextInLanguage(text, languageConfig) {
  const value = typeof text === "string" ? text.trim() : "";
  if (!value) return false;

  const hebrewCount = countMatches(value, HEBREW_REGEX);
  const latinCount = countMatches(value, LATIN_REGEX);

  if (languageConfig.key === "en") {
    if (hebrewCount > 0) return false;
    return latinCount > 0;
  }

  if (languageConfig.key === "he") {
    if (latinCount > 0) return false;
    return hebrewCount > 0;
  }

  return true;
}

function getHardLanguageFallback(languageConfig) {
  if (languageConfig.key === "he") {
    return "אני יכול לענות רק בעברית בשיחה הזאת.";
  }
  return "I can answer only in English in this session.";
}

function getLanguageScriptRule(languageConfig) {
  if (languageConfig.key === "he") {
    return "Use Hebrew letters only. Never use Latin letters A-Z.";
  }
  return "Use English letters only. Never use Hebrew letters (U+0590-U+05FF).";
}

process.on("unhandledRejection", (reason) => {
  logError("Unhandled Promise Rejection", reason);
});

process.on("uncaughtException", (error) => {
  logError("Uncaught Exception", error);
});

app.use((req, _res, next) => {
  console.log(`[${new Date().toISOString()}] ${req.method} ${req.originalUrl}`);
  next();
});
app.use((req, res, next) => {
  res.on("finish", () => {
    if (res.statusCode >= 400) {
      console.error(
        `[${new Date().toISOString()}] HTTP ${res.statusCode} ${req.method} ${req.originalUrl}`
      );
    }
  });
  next();
});

const webRoot = process.cwd();
app.get("/", (_req, res) => {
  return res.sendFile(join(webRoot, "index.html"));
});
app.get("/app.js", (_req, res) => {
  return res.sendFile(join(webRoot, "app.js"));
});
app.get("/styles.css", (_req, res) => {
  return res.sendFile(join(webRoot, "styles.css"));
});

if (!OPENAI_API_KEY) {
  console.warn("Missing OPENAI_API_KEY in environment.");
}
if (!GOOGLE_CLOUD_PROJECT_ID) {
  console.warn("GOOGLE_CLOUD_PROJECT_ID not set. Continuing without x-goog-user-project header.");
}
console.log(`Google auth source: ${GOOGLE_AUTH_SOURCE}`);
console.log(`Supported languages: ${Object.keys(LANGUAGE_CONFIG).join(", ")}`);

async function transcribeWithOpenAI(audioBuffer, mimeType, languageConfig) {
  const formData = new FormData();
  formData.append(
    "file",
    new Blob([audioBuffer], { type: mimeType || "audio/webm" }),
    "speech.webm"
  );
  formData.append("model", TRANSCRIBE_MODEL);
  if (languageConfig?.sttCode) {
    formData.append("language", languageConfig.sttCode);
  }

  const response = await fetch("https://api.openai.com/v1/audio/transcriptions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${OPENAI_API_KEY}`,
    },
    body: formData,
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Transcription failed: ${errorText}`);
  }

  const data = await response.json();
  return data.text?.trim() || "";
}

function normalizeHistory(historyInput, languageConfig) {
  if (!Array.isArray(historyInput)) return [];

  return historyInput
    .filter(
      (item) =>
        item &&
        (item.role === "user" || item.role === "assistant") &&
        typeof item.content === "string" &&
        item.content.trim().length > 0
    )
    .slice(-12)
    .map((item) => ({ role: item.role, content: item.content.trim() }))
    .filter((item) => {
      if (item.role === "user") return true;
      if (!languageConfig) return true;
      return isTextInLanguage(item.content, languageConfig);
    });
}

async function requestChatCompletion(messages, model, temperature, maxTokens) {
  const response = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${OPENAI_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model,
      messages,
      temperature,
      max_tokens: maxTokens,
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`LLM call failed: ${errorText}`);
  }

  const data = await response.json();
  return data.choices?.[0]?.message?.content?.trim() || "";
}

async function rewriteReplyToLanguage(rawReply, languageConfig, fast) {
  const rewriteModel = fast ? FAST_LLM_MODEL : DEFAULT_LLM_MODEL;
  const scriptRule = getLanguageScriptRule(languageConfig);
  const rewritten = await requestChatCompletion(
    [
      {
        role: "system",
        content: `You are a strict language rewriter. Rewrite the text only in ${languageConfig.label}. ${scriptRule} Keep the same meaning. Output only the rewritten sentence.`,
      },
      {
        role: "user",
        content: rawReply || "",
      },
    ],
    rewriteModel,
    0,
    Math.max(60, FAST_MAX_TOKENS)
  );

  return rewritten.trim();
}

async function chatWithOpenAI(userText, historyInput, languageConfig, options = {}) {
  const fast = Boolean(options.fast);
  const history = normalizeHistory(historyInput, languageConfig);
  const model = fast ? FAST_LLM_MODEL : DEFAULT_LLM_MODEL;
  const scriptRule = getLanguageScriptRule(languageConfig);
  const systemPrompt = fast
    ? `You are a real-time voice assistant. Reply only in ${languageConfig.label}. ${scriptRule} Never switch language, even if asked. If user asks to change language, refuse in ${languageConfig.label}. Keep it one short sentence under 12 words.`
    : `You are a concise and helpful voice assistant. Reply only in ${languageConfig.label}. ${scriptRule} Never switch language during this conversation, even if asked. If user asks to change language, refuse in ${languageConfig.label}.`;

  const rawReply = await requestChatCompletion(
    [
      { role: "system", content: systemPrompt },
      ...history,
      { role: "user", content: userText },
    ],
    model,
    fast ? 0.2 : 0.3,
    fast ? FAST_MAX_TOKENS : 180
  );

  let finalReply = rawReply || getHardLanguageFallback(languageConfig);
  if (!isTextInLanguage(finalReply, languageConfig)) {
    console.error(
      `[language-guard] Off-language reply detected for ${languageConfig.key}. Rewriting response.`
    );
    try {
      const rewritten = await rewriteReplyToLanguage(finalReply, languageConfig, fast);
      if (isTextInLanguage(rewritten, languageConfig)) {
        finalReply = rewritten;
      } else {
        console.error(
          `[language-guard] Rewrite failed for ${languageConfig.key}. Using hard fallback response.`
        );
        finalReply = getHardLanguageFallback(languageConfig);
      }
    } catch (rewriteError) {
      logError("Language rewrite failed", rewriteError, { language: languageConfig.key });
      finalReply = getHardLanguageFallback(languageConfig);
    }
  }

  return finalReply;
}

async function synthesizeWithGoogleChirp(text, languageConfig) {
  const authOptions = {
    scopes: ["https://www.googleapis.com/auth/cloud-platform"],
  };
  if (GOOGLE_CREDENTIALS) {
    authOptions.credentials = GOOGLE_CREDENTIALS;
  }
  const auth = new GoogleAuth(authOptions);

  const client = await auth.getClient();
  const tokenResult = await client.getAccessToken();
  const accessToken = typeof tokenResult === "string" ? tokenResult : tokenResult?.token;

  if (!accessToken) {
    throw new Error("Could not get Google access token.");
  }

  const headers = {
    Authorization: `Bearer ${accessToken}`,
    "Content-Type": "application/json",
  };
  if (GOOGLE_CLOUD_PROJECT_ID) {
    headers["x-goog-user-project"] = GOOGLE_CLOUD_PROJECT_ID;
  }

  const response = await fetch(GOOGLE_TTS_ENDPOINT, {
    method: "POST",
    headers,
    body: JSON.stringify({
      input: { text },
      voice: {
        languageCode: languageConfig.chirpLanguageCode,
        name: languageConfig.chirpVoice,
      },
      audioConfig: {
        audioEncoding: "MP3",
      },
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Google TTS failed: ${errorText}`);
  }

  const data = await response.json();
  return data.audioContent;
}

app.post("/api/client-error", (req, res) => {
  console.error(`\n[${new Date().toISOString()}] Browser Error Report`);
  console.error(req.body);
  return res.json({ ok: true });
});
app.all("/api/client-error", (_req, res) => {
  return res.status(405).json({ error: "Method not allowed. Use POST /api/client-error." });
});

app.post("/api/voice", upload.single("audio"), async (req, res) => {
  try {
    const totalStart = performance.now();
    if (!OPENAI_API_KEY) {
      return res.status(500).json({
        error: "Server env missing. Set OPENAI_API_KEY before running.",
      });
    }

    if (!req.file?.buffer) {
      return res.status(400).json({ error: "Audio file is required." });
    }

    const languageKey = resolveConversationLanguage(req.body?.language);
    const languageConfig = LANGUAGE_CONFIG[languageKey];

    const transcribeStart = performance.now();
    const transcript = await transcribeWithOpenAI(req.file.buffer, req.file.mimetype, languageConfig);
    const transcribeMs = Math.round(performance.now() - transcribeStart);
    if (!transcript) {
      return res.status(400).json({ error: "Could not transcribe speech." });
    }

    let history = [];
    if (typeof req.body?.history === "string" && req.body.history.trim()) {
      try {
        history = JSON.parse(req.body.history);
      } catch (parseError) {
        logError("History parse failed", parseError, { rawHistory: req.body.history.slice(0, 300) });
      }
    }

    const preferFast = req.body?.fast === "1" || FAST_RESPONSE_MODE;
    const llmModelUsed = preferFast ? FAST_LLM_MODEL : DEFAULT_LLM_MODEL;
    const chirpAvailable = Boolean(languageConfig.chirpVoice);

    const llmStart = performance.now();
    const reply = await chatWithOpenAI(transcript, history, languageConfig, { fast: preferFast });
    const llmMs = Math.round(performance.now() - llmStart);

    let audioBase64 = null;
    let audioMime = null;
    let ttsMs = 0;
    let fastTts = preferFast || !chirpAvailable;
    const ttsModelUsed = fastTts ? null : languageConfig.chirpVoice;

    if (!fastTts) {
      const ttsStart = performance.now();
      audioBase64 = await synthesizeWithGoogleChirp(reply, languageConfig);
      ttsMs = Math.round(performance.now() - ttsStart);
      audioMime = "audio/mp3";
    }

    const totalMs = Math.round(performance.now() - totalStart);
    console.log(
      `[latency] lang=${languageKey} transcribe=${transcribeMs}ms llm=${llmMs}ms tts=${ttsMs}ms total=${totalMs}ms fast=${fastTts}`
    );

    return res.json({
      transcript,
      reply,
      audioBase64,
      audioMime,
      fastTts,
      language: languageKey,
      models: {
        transcribe: `${TRANSCRIBE_MODEL} (${languageConfig.sttCode})`,
        llm: `${llmModelUsed} (${languageConfig.key})`,
        tts: ttsModelUsed,
      },
      timings: {
        transcribeMs,
        llmMs,
        ttsMs,
        totalMs,
      },
    });
  } catch (error) {
    logError("API /api/voice failed", error, {
      mimeType: req.file?.mimetype || null,
      size: req.file?.size || null,
    });
    return res.status(500).json({ error: error.message || "Something went wrong." });
  }
});
app.all("/api/voice", (_req, res) => {
  return res.status(405).json({ error: "Method not allowed. Use POST /api/voice." });
});
app.get("/api/health", (_req, res) => {
  return res.json({ ok: true, port });
});
app.use("/api", (_req, res) => {
  return res.status(404).json({ error: "API route not found." });
});

app.use((error, req, res, next) => {
  logError(`Express middleware error on ${req.method} ${req.originalUrl}`, error);
  if (res.headersSent) {
    return next(error);
  }
  return res.status(500).json({ error: "Internal server error." });
});

if (!process.env.VERCEL) {
  const server = app.listen(port, () => {
    console.log(`Voice agent running on http://localhost:${port}`);
  });

  server.on("error", (error) => {
    logError(`Server listen error on port ${port}`, error);
  });
}

export default app;

