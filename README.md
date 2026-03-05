# AI Production Scheduler v3

An interactive web-based production scheduling optimizer powered by **Google OR-Tools CP-SAT** with an integrated **AI Assistant** (Azure OpenAI).

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the application
python app.py

# 3. Open in browser
http://localhost:5050
```

## Features

### Core Scheduler
- **Multi-Week Planning**: Load production orders grouped by Exit Factory Year & Week
- **CP-SAT Optimizer**: Constraint Programming solver for optimal daily scheduling
- **Advanced Constraints**:
  - Rear Loader 2512: Max 1 per day, evenly spaced across the week
  - PKRRLSB (Rear Loader): Prefer Tuesday / Thursday
  - PKRML (Manual Side Loader): Prefer Tuesday / Thursday
  - DC Refuse Stock: Prefer Friday
- **Interactive Dashboard**: KPI cards, stacked bar charts, heatmap, compliance panel

### AI Assistant (New in v3)
- **Context-Aware Chat**: Understands loaded data, solver results, and constraints
- **Azure OpenAI Integration**: Connect your corporate Azure OpenAI API
- **Daily Token Budget Control**: Set daily limits to manage LLM costs
- **Quick Suggestions**: Pre-built questions for common queries

## AI Configuration

### Option 1: Via the UI
1. Click the robot icon (bottom-right corner)
2. Click the gear icon (Settings)
3. Enter your Azure OpenAI credentials:
   - **Endpoint URL**: `https://your-resource.openai.azure.com/`
   - **API Key**: Your Azure OpenAI API key
   - **API Version**: e.g., `2024-12-01-preview`
   - **Deployment Name**: e.g., `gpt-4o`
4. Set token budget limits
5. Click "Save Configuration"

### Option 2: Via Environment Variables
```bash
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_DEPLOYMENT="gpt-4o"
export AZURE_OPENAI_API_VERSION="2024-12-01-preview"
python app.py
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve the web UI |
| `/weeks` | GET | List available production weeks |
| `/week_data/<key>` | GET | Get demand rows for a specific week |
| `/solve` | POST | Run the CP-SAT optimizer |
| `/ai/config` | GET | Get AI configuration (key masked) |
| `/ai/config` | POST | Update AI configuration |
| `/ai/usage` | GET | Get today's token usage |
| `/ai/usage/history` | GET | Get full token usage history |
| `/ai/chat` | POST | Send a message to the AI assistant |

## Token Budget Control

The system tracks daily token consumption and enforces limits:
- **Daily Token Limit**: Maximum tokens (prompt + completion) per day
- **Max Tokens Per Reply**: Cap on each individual AI response
- Usage resets at midnight
- Visual progress bar in the chat panel shows real-time consumption

## Project Structure

```
scheduler_demo/
├── app.py                 # Flask backend + CP-SAT solver + AI chat
├── requirements.txt       # Python dependencies
├── static/
│   └── index.html         # Interactive frontend (single-file)
├── data/
│   └── daily_tactical_scheduler_v2_2_2026.csv  # Production data
├── ai_config.json         # AI settings (auto-created)
└── token_usage.json       # Daily token log (auto-created)
```
