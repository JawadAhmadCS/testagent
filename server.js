import "dotenv/config";
import express from "express";
import multer from "multer";
import { GoogleAuth } from "google-auth-library";
import { performance } from "node:perf_hooks";

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
const CHIRP_VOICE = process.env.CHIRP_VOICE || "en-US-Chirp3-HD-Kore";
const GOOGLE_TTS_ENDPOINT =
  process.env.GOOGLE_TTS_ENDPOINT || "https://texttospeech.googleapis.com/v1/text:synthesize";
const TRANSCRIBE_MODEL = process.env.TRANSCRIBE_MODEL || "gpt-4o-transcribe";
const DEFAULT_LLM_MODEL = process.env.DEFAULT_LLM_MODEL || "gpt-4o";
const FAST_LLM_MODEL = process.env.FAST_LLM_MODEL || "gpt-4o-mini";
const FAST_RESPONSE_MODE = process.env.FAST_RESPONSE_MODE !== "0";
const FAST_MAX_TOKENS = Number(process.env.FAST_MAX_TOKENS || 48);

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

if (!OPENAI_API_KEY) {
  console.warn("Missing OPENAI_API_KEY in environment.");
}
if (!GOOGLE_CLOUD_PROJECT_ID) {
  console.warn("GOOGLE_CLOUD_PROJECT_ID not set. Continuing without x-goog-user-project header.");
}
console.log(`Google auth source: ${GOOGLE_AUTH_SOURCE}`);

async function transcribeWithOpenAI(audioBuffer, mimeType) {
  const formData = new FormData();
  formData.append(
    "file",
    new Blob([audioBuffer], { type: mimeType || "audio/webm" }),
    "speech.webm"
  );
  formData.append("model", TRANSCRIBE_MODEL);

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

function normalizeHistory(historyInput) {
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
    .map((item) => ({ role: item.role, content: item.content.trim() }));
}

async function chatWithOpenAI(userText, historyInput, options = {}) {
  const fast = Boolean(options.fast);
  const history = normalizeHistory(historyInput);
  const model = fast ? FAST_LLM_MODEL : DEFAULT_LLM_MODEL;
  const systemPrompt = fast
    ? "You are a real-time voice assistant. Reply in one short sentence, under 12 words."
    : "You are a concise and helpful voice assistant. Keep answers short and clear.";
  const response = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${OPENAI_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model,
      messages: [
        { role: "system", content: systemPrompt },
        ...history,
        { role: "user", content: userText },
      ],
      temperature: fast ? 0.2 : 0.7,
      max_tokens: fast ? FAST_MAX_TOKENS : 180,
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`LLM call failed: ${errorText}`);
  }

  const data = await response.json();
  return data.choices?.[0]?.message?.content?.trim() || "Sorry, I could not respond.";
}

async function synthesizeWithGoogleChirp(text) {
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
        languageCode: "en-US",
        name: CHIRP_VOICE,
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

    const transcribeStart = performance.now();
    const transcript = await transcribeWithOpenAI(req.file.buffer, req.file.mimetype);
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
    const ttsModelUsed = preferFast ? null : CHIRP_VOICE;

    const llmStart = performance.now();
    const reply = await chatWithOpenAI(transcript, history, { fast: preferFast });
    const llmMs = Math.round(performance.now() - llmStart);

    let audioBase64 = null;
    let audioMime = null;
    let ttsMs = 0;
    let fastTts = preferFast;

    if (!preferFast) {
      const ttsStart = performance.now();
      audioBase64 = await synthesizeWithGoogleChirp(reply);
      ttsMs = Math.round(performance.now() - ttsStart);
      audioMime = "audio/mp3";
      fastTts = false;
    }

    const totalMs = Math.round(performance.now() - totalStart);
    console.log(
      `[latency] transcribe=${transcribeMs}ms llm=${llmMs}ms tts=${ttsMs}ms total=${totalMs}ms fast=${fastTts}`
    );

    return res.json({
      transcript,
      reply,
      audioBase64,
      audioMime,
      fastTts,
      models: {
        transcribe: TRANSCRIBE_MODEL,
        llm: llmModelUsed,
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

const server = app.listen(port, () => {
  console.log(`Voice agent running on http://localhost:${port}`);
});

server.on("error", (error) => {
  logError(`Server listen error on port ${port}`, error);
});

