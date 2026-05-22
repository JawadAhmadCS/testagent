# testagent
# voiceflow-realestate

## Run Project (PowerShell)

1. Go to project folder:
```powershell
cd "c:\Users\mjawa\Pictures\AI Projects\testagent"
```

2. Create env file from example:
```powershell
Copy-Item .env.example .env
```

3. Open `.env` and set required values:
- `OPENAI_API_KEY` is required
- For Google TTS also set:
  - `GOOGLE_CLOUD_PROJECT_ID`
  - `GOOGLE_SERVICE_ACCOUNT_B64`
- Optional for Hebrew voice dropdown:
  - `CHIRP_VOICES_HE` (comma-separated list of Hebrew Google voice IDs)

4. Install dependencies:
```powershell
npm install
```

5. Start server:
```powershell
npm start
```

6. Open in browser:
```text
http://localhost:3000
```

## Notes

- If `OPENAI_API_KEY` is missing, `/api/voice` will fail.
- `server.js` is required for mic/voice features because frontend calls `/api/*` routes from backend.
