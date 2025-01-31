# urmom-bot

A Discord bot that responds with "ur mom" jokes when messages are reacted to with a clown emoji (🤡).

## Features
- Generates AI-powered "ur mom" jokes in response to 🤡 emoji reactions
- Creates inappropriate culturally-themed jokes when triggered by country flag emoji reactions
- Powered by Google's Gemini AI

## Adaptive Learning
The bot features an adaptive learning system that improves its joke generation over time:

- Collects and stores jokes (both "ur mom" and "in Soviet Russia" variants)
- Learns from both AI-generated and user-contributed jokes
- Evaluates joke quality based on user reactions
- Uses popular jokes as reference material for generating new content

For a joke to be considered for learning (both bot-generated and user-contributed):
- Must be a reply to the original message that inspired the joke
- Must follow "ur mom" joke pattern
- Must receive reactions from server members

The learning system prioritizes jokes with higher engagement (more reactions) when creating new content, ensuring better quality over time.

## Configuration

Create a `.env` file in the root directory with the following variables:

```env
DISCORD_TOKEN=      # Your Discord bot token
GEMINI_API_KEY=     # Your Gemini API key
GEMINI_TEMPERATURE= # Model temperature (recommended: 1.0)
GEMINI_MODEL=       # Model name (e.g. "gemini-exp-1206")
```

## Server Setup
To ensure proper bot operation:
- Create a channel named `#bots` - the bot will post all generated jokes there
- The bot will react to emoji triggers in any channel, but responses are always posted in `#bots`

## Where to get the keys:
- Discord token: [Discord Developer Portal](https://discord.com/developers/applications)
- Gemini API key: [Google AI Studio](https://aistudio.google.com)

## Required Bot Permissions
1. In Discord Developer Portal:
   - Go to Settings > Bot
   - Enable "Message Content Intent"
2. In Settings > Installation:
   - Select "bot" under Scopes
   - Enable "Send Messages" permission

## Running the Bot

Simply run:
```bash
docker compose up -d
```

To view logs:
```bash
docker compose logs -f
```

## Try It Out! 🤖
Want to test the bot without setting up your own instance? You can add my hosted instance to your Discord server:

[➡️ Add Bot to Your Server](https://discord.com/oauth2/authorize?client_id=1333878858138652682)