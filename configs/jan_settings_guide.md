# Connecting DEVIOS Turbo to Jan (jan.ai)

Jan v0.8.x supports custom OpenAI-compatible local providers. Follow these steps to route all requests to the high-performance DEVIOS Turbo backend.

## Quick Setup via UI

1. Start the DEVIOS Turbo server:
   ```cmd
   scripts\run_server.bat
   ```
2. Open **Jan** (`Jan-Desktop.exe`).
3. Click the **Gear icon (Settings)** at the bottom-left.
4. Select **Model Providers** -> **OpenAI** (or **Custom Provider**):
   - **Active**: On (`true`)
   - **Base URL**: `http://127.0.0.1:18080/v1`
   - **API Key**: `devios-turbo` (any non-empty placeholder)
5. Under models, select `Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Turbo` (or `devios-turbo`).
6. Start a new chat. Your queries will stream through the speculative decoding engine at peak memory bandwidth.

## Advanced Setup via `%APPDATA%\Jan\data\settings.json`

To inject the configuration automatically into Jan's configuration file, configure the `openai` provider block in `settings.json`:

```json
{
  "provider": "openai",
  "active": true,
  "base_url": "http://127.0.0.1:18080/v1",
  "settings": [
    {
      "key": "api-key",
      "controller_props": { "value": "devios-turbo" }
    },
    {
      "key": "base-url",
      "controller_props": { "value": "http://127.0.0.1:18080/v1" }
    }
  ],
  "models": [
    {
      "id": "Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Turbo",
      "name": "DEVIOS Turbo: Gemma 4B Aggressive",
      "capabilities": ["completion", "tools"]
    },
    {
      "id": "devios-turbo",
      "name": "DEVIOS Turbo (Default)",
      "capabilities": ["completion", "tools"]
    }
  ]
}
```
