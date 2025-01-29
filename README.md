# urmom-bot

A Discord bot that responds with "ur mom" jokes when messages are reacted to with a clown emoji (🤡).

## Features
- Generates custom "ur mom" jokes using AI
- Triggered by clown emoji reactions
- Powered by Google's Gemini AI

## Configuration

Create a `.env` file in the root directory with the following variables:

```env
DISCORD_TOKEN=      # Your Discord bot token
GEMINI_API_KEY=     # Your Gemini API key
GEMINI_TEMPERATURE= # Model temperature (recommended: 1.0)
GEMINI_MODEL=       # Model name (e.g. "gemini-exp-1206")
```

### Where to get the keys:
- Discord token: [Discord Developer Portal](https://discord.com/developers/applications)
- Gemini API key: [Google AI Studio](https://aistudio.google.com)

## Running the Bot

Simply run:
```bash
docker compose up -d
```

To view logs:
```bash
docker compose logs -f
```