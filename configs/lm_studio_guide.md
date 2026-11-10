# Connecting DEVIOS Turbo to LM Studio

Use LM Studio's graphical user interface as a frontend for the local DEVIOS Turbo server.

## Step 1. Start Server
Run the background server on port `18080`:
```cmd
scripts\run_server.bat
```

## Step 2. Add Provider in LM Studio
1. Open **LM Studio**.
2. Go to **Settings** -> **Providers** (or Custom / OpenAI-compatible Providers).
3. Click **Add Provider**:
   - **Provider Name**: `DEVIOS Turbo`
   - **Base URL**: `http://127.0.0.1:18080/v1`
   - **API Key**: `devios`
4. Click **Test Connection**. LM Studio queries `GET /v1/models` and discovers the loaded GGUF model.

## Step 3. Chat with Real-Time Streaming
1. Open the **Chat** tab in LM Studio.
2. Select `Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Turbo` from the model dropdown.
3. Chat with real-time SSE token streaming and instant stop button support.
