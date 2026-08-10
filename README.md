# urmom-bot / мамкин бот

A Discord bot that responds with "ur mom" jokes when messages are reacted to with a clown emoji (🤡).

## Features
- **Memory System**: Remembers facts about users and provides personalized responses
- **Transient Memory**: Passively learns about users and chat dynamics from ongoing conversations
- **General AI Assistant**: Answers questions as any AI does
- **Celebrity Impersonation**: Generates responses as famous people with their unique style and personality
- **AI-Powered Jokes**: Creates "ur mom" jokes (🤡) and culturally-themed humor (flag emoji reactions)
- **Street Wisdom**: Generates witty one-liner observations from chat context (🧔 or 🧠)
- **Devil's Advocate**: Challenges opinions with counter-arguments (😈)
- **Video Auto-Embedding**: Extracts and embeds videos from X/Twitter, Instagram Reels, and Reddit posts, with ffmpeg compression for oversized videos
- **Self-Cleaning**: Auto-archives jokes, removes them after a set time, or when downvoted by the community
- **Multi-Language Support**: Works in any language (English, Russian, French, Japanese, etc.)
- **Scheduled Tasks**: Create recurring or one-off AI tasks in plain language — daily reports, reminders, digests
- **Multiple AI Providers**: Uses different AI models for different tasks with automatic fallback
  - Claude: Analytical tasks
  - Gemini Flash: General queries and information retrieval
  - Grok: Creative tasks
  - Gemma: Language detection and structured operations

## Adaptive Learning
The bot features an adaptive learning system that improves its joke generation over time:

- Learns from both AI-generated and user-contributed "ur mom" jokes
- Evaluates joke quality based on user reactions
- Uses popular jokes as reference material for generating new content

For a joke to be considered for learning (both bot-generated and user-contributed):
- Must be a reply to the original message that inspired the joke
- Must receive reactions from server members
- AI must confirm it's actually a joke

The learning system prioritizes jokes with higher engagement (more reactions) when creating new content, ensuring better quality over time.

## Server Setup

### Environment Variables
Create a `.env` file in the root directory with the following variables:

```env
# Discord Configuration (Required)
DISCORD_TOKEN=your_discord_bot_token      # Get from Discord Developer Portal

# Gemini Configuration (Required)
GEMINI_API_KEY=your_gemini_api_key        # Get from Google AI Studio
GEMINI_FLASH_MODEL=gemini-2.5-flash      # Flash model name

# Gemma Configuration (Required)
GEMMA_API_KEY=your_gemma_api_key          # Get from Google AI Studio (separate key)
GEMMA_MODEL=gemma-3-27b-it                # Gemma model name

# Grok Configuration (Required)
GROK_API_KEY=your_grok_api_key            # Get from xAI platform
GROK_MODEL=grok-3-mini                    # Grok model name

# TinyURL Configuration (Optional - fallback for very large videos)
TINYURL_API_TOKEN=your_tinyurl_api_token  # Get from TinyURL API Settings
```

### Where to get the keys:
- Discord token: [Discord Developer Portal](https://discord.com/developers/applications)
- Gemini API key: [Google AI Studio](https://aistudio.google.com)
- Grok API key: [xAI Platform](https://console.x.ai/)
- TinyURL API token: [TinyURL API Settings](https://tinyurl.com/app/settings/api)

**Note**: All AI provider keys are required for full functionality.

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

## Memory Commands
The bot can remember and forget facts about users:
- `@urmom-bot remember that @John likes pizza` - Store a fact about a user
- `@urmom-bot forget that @John likes pizza` - Remove a specific fact

## Bot Configuration
The following commands are available:
- `@urmom-bot help` - Show info on available commands
- `@urmom-bot settings` - Display current configuration
- `@urmom-bot setArchiveChannel #bot-jokes` - Set channel name for bot jokes, empty to disable
- `@urmom-bot deleteJokesAfterMinutes X` - 0 for disabled, otherwise bot will delete jokes after X minutes
- `@urmom-bot deleteJokesWhenDownvoted X` - Delete jokes if downvotes - upvotes >= X, 0 to disable
- `@urmom-bot enableCountryJokes true/false` - Enable/disable country-specific jokes
- `@urmom-bot setDefaultTimezone Asia/Tokyo` - Set default IANA timezone for scheduled tasks

## Bot Behavior
What can this bot do?

### General AI Assistant
- **Answer any question**: `@urmom-bot explain quantum physics`
- **Query memories**: `@urmom-bot what do you remember about John?`
- **Choose AI backend**: `@urmom-bot ask grok about creative writing` or `@urmom-bot have codex explain this code`

### Scheduled Tasks
Ask the bot to run any prompt on a schedule, in plain language — recurring or one-off. The task fires in the channel where it was created:
- `@urmom-bot set up a daily task to report on semiconductor stocks after NYSE close`
- `@urmom-bot set up a daily task to read Trump's tweets — who is he threatening next?`
- `@urmom-bot remind me tomorrow at 3pm to actually go outside`

Manage existing tasks the same way: `what's scheduled here?`, `change task 5 to run at 8am`, `run task 5 now`, `delete task 5`. Scheduling is timezone-aware — say "9am Tokyo time" and it just works; if no timezone is mentioned, the guild's default applies (see `setDefaultTimezone`).

### Celebrity Impersonation
- Respond as famous personalities with `@urmom-bot what would <famous person> say?`
  - Examples:
    - `@urmom-bot what would Jesus say if we could rap like Eminen?`
    - `@urmom-bot what would Trump say if he was a software developer?`
    - `@urmom-bot Что бы сказал Гоблин если бы он делал свой перевод?`

## Try It Out! 🤖
Want to test the bot without setting up your own instance? You can add my hosted instance to your Discord server:

[➡️ Add Bot to Your Server](https://discord.com/oauth2/authorize?client_id=1333878858138652682)