# urmom-bot / мамкин бот

A Discord bot that responds with "ur mom" jokes when messages are reacted to with a clown emoji (🤡).

## Features
- Generates AI-powered "ur mom" jokes in response to 🤡 emoji reactions
- Creates inappropriate culturally-themed jokes when triggered by country flag emoji reactions
- Powered by Google's Gemini (free) and Grok (paid) for enhanced joke generation

## Adaptive Learning
The bot features an adaptive learning system that improves its joke generation over time:

- Learns from both AI-generated and user-contributed "ur mom" jokes
- Evaluates joke quality based on user reactions
- Uses popular jokes as reference material for generating new content

For a joke to be considered for learning (both bot-generated and user-contributed):
- Must be a reply to the original message that inspired the joke
- Must follow "ur mom" joke pattern
- Must receive reactions from server members

The learning system prioritizes jokes with higher engagement (more reactions) when creating new content, ensuring better quality over time.

## Server Setup

### Environment Variables
Create a `.env` file in the root directory with the following variables:

```env
# Select AI Provider (Required)
AI_PROVIDER=GEMINI                        # Choose either GEMINI or GROK

# Discord Configuration (Required)
DISCORD_TOKEN=your_discord_bot_token      # Get from Discord Developer Portal

# Gemini Configuration (Required if AI_PROVIDER=GEMINI)
GEMINI_API_KEY=your_gemini_api_key        # Get from Google AI Studio
GEMINI_TEMPERATURE=0.7                    # Value between 0-2, higher = more creative
GEMINI_MODEL=gemini-2.0-flash-exp         # Latest experimental model for best results

# Grok Configuration (Required if AI_PROVIDER=GROK)
GROK_API_KEY=your_grok_api_key            # Get from xAI platform
GROK_TEMPERATURE=0.7                      # Value between 0-2, higher = more creative
GROK_MODEL=grok-2-1212                    # Latest model version
```

### Where to get the keys:
- Discord token: [Discord Developer Portal](https://discord.com/developers/applications)
- Gemini API key: [Google AI Studio](https://aistudio.google.com)

### Required Bot Permissions
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

## Bot Configuration
The following commands are available:
- `@urmom-bot help` - Show info on available commands
- `@urmom-bot settings` - Display current configuration
- `@urmom-bot setArchiveChannel #bot-jokes` - Set channel name for bot jokes, empty to disable
- `@urmom-bot deleteJokesAfterMinutes X` - 0 for disabled, otherwise bot will delete jokes after X minutes
- `@urmom-bot deleteJokesWhenDownvoted X` - Delete jokes if downvotes - upvotes >= X, 0 to disable
- `@urmom-bot enableCountryJokes true/false` - Enable/disable country-specific jokes

## Bot Behavior
What can this bot do?
- Generate "ur mom" jokes when someone reacts to a message with 🤡
- Create country-specific jokes when someone uses a flag emoji reaction
- Archive all jokes to a dedicated channel of your choice
- Keep your chat channels clean by automatically removing joke responses after a set time
- Listen to your community's feedback - if a joke gets too many 👎 reactions, it gets removed

## Try It Out! 🤖
Want to test the bot without setting up your own instance? You can add my hosted instance to your Discord server:

[➡️ Add Bot to Your Server](https://discord.com/oauth2/authorize?client_id=1333878858138652682)