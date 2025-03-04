import asyncio
import datetime
import os
from enum import Enum
from types import SimpleNamespace

import nextcord
from dotenv import load_dotenv
from langdetect import detect, LangDetectException

from container import container  # Import the instance instead of the class

load_dotenv()

intents = nextcord.Intents.default()
intents.message_content = True  # MUST have this to receive message content

bot = nextcord.Client(intents=intents)

# Create cache object to hold bot state
cache = SimpleNamespace()
cache.processed_messages = set()
cache.joke_cache = {}  # message_id -> bool

class BotCommand(Enum):
    HELP = "help"
    SETTINGS = "settings"
    SET_ARCHIVE_CHANNEL = "setarchivechannel"
    DELETE_JOKES_AFTER = "deletejokesafterminutes"
    DELETE_JOKES_WHEN_DOWNVOTED = "deletejokeswhendownvoted"
    ENABLE_COUNTRY_JOKES = "enablecountryjokes"

    @classmethod
    def from_str(cls, value: str) -> 'BotCommand | None':
        try:
            return next(cmd for cmd in cls if cmd.value == value.lower())
        except StopIteration:
            return None

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user}')

@bot.event
async def on_raw_reaction_add(payload):
    try:
        emoji_str = str(payload.emoji)
        country = await container.country_resolver.get_country_from_flag(emoji_str)
        message_key = (payload.message_id, emoji_str)
        
        is_clown = emoji_str == "🤡"
        is_country = country is not None
        is_thumbs_down = emoji_str == "👎"

        if (is_clown or is_country) and message_key not in cache.processed_messages:
            cache.processed_messages.add(message_key)
            await process_joke_request(payload, country)
        elif is_thumbs_down:
            await retract_joke(payload)
        elif await is_joke(payload):
            await save_joke(payload)
    except ValueError as e:
        print(f"Error processing reaction: {e}")

@bot.event
async def on_message(message: nextcord.Message):
    container.telemetry.increment_message_counter(message)
    
    if message.author.bot:
        return
    
    if not message.content.startswith(f"<@{bot.user.id}>"):
        return

    extracted_message = message.content.replace(f"<@{bot.user.id}>", "").strip()
    famous_person = await container.ai_client.is_famous_person_request(extracted_message)
    
    if famous_person:
        await process_famous_person_query(message, famous_person)
        return

    # Check if user is admin before processing commands
    if not message.author.guild_permissions.administrator:
        await message.reply("Sorry, only administrators can use bot commands!")
        return

    args = message.content.split()[1:]
    if not args:
        return

    command = BotCommand.from_str(args[0])
    if not command:
        return
        
    config = container.store.get_guild_config(message.guild.id)

    if command == BotCommand.HELP:
        help_text = """
Available commands:
`@urmom-bot settings` - Show current configuration
`@urmom-bot setArchiveChannel #channel` - Set channel for permanent joke storage
`@urmom-bot deleteJokesAfterMinutes X` - Delete jokes after X minutes (0 to disable)
`@urmom-bot deleteJokesWhenDownvoted X` - Delete jokes when downvotes - upvotes >= X (0 to disable)
`@urmom-bot enableCountryJokes true/false` - Enable/disable country-specific jokes
"""
        await message.reply(help_text)
        return

    if command == BotCommand.SETTINGS:
        settings_text = f"""
Current settings:
• Archive Channel: {f'<#{config.archive_channel_id}>' if config.archive_channel_id else 'Disabled'}
• Auto-delete after: {config.delete_jokes_after_minutes} minutes (0 = never)
• Delete on downvotes: {config.downvote_reaction_threshold} (0 = disabled)
• Country jokes: {'Enabled' if config.enable_country_jokes else 'Disabled'}
"""
        await message.reply(settings_text)
        return

    if command == BotCommand.SET_ARCHIVE_CHANNEL:
        if not message.channel_mentions:
            config.archive_channel_id = 0  # Disable archiving
            container.store.save_guild_config(config)
            await message.reply("Joke archiving has been disabled.")
            return
            
        config.archive_channel_id = message.channel_mentions[0].id
        container.store.save_guild_config(config)
        await message.reply(f"Jokes will now be archived in {message.channel_mentions[0].mention}")
        return

    elif command == BotCommand.DELETE_JOKES_AFTER:
        try:
            minutes = int(args[1])
            if minutes < 0:
                raise ValueError
            config.delete_jokes_after_minutes = minutes
            container.store.save_guild_config(config)
            await message.reply(f"Jokes will be deleted after {minutes} minutes (0 = never)")
        except (IndexError, ValueError):
            await message.reply("Please provide a valid number of minutes!")

    elif command == BotCommand.DELETE_JOKES_WHEN_DOWNVOTED:
        try:
            threshold = int(args[1])
            if threshold < 0:
                raise ValueError
            config.downvote_reaction_threshold = threshold
            container.store.save_guild_config(config)
            await message.reply(f"Jokes will be deleted when downvotes - upvotes >= {threshold}")
        except (IndexError, ValueError):
            await message.reply("Please provide a valid threshold number!")

    elif command == BotCommand.ENABLE_COUNTRY_JOKES:
        try:
            enable = args[1].lower() == "true"
            config.enable_country_jokes = enable
            container.store.save_guild_config(config)
            await message.reply(f"Country jokes {'enabled' if enable else 'disabled'}")
        except IndexError:
            await message.reply("Please specify true or false!")

async def process_famous_person_query(message, famous_person):
    try:
        reference_message = None
        if message.reference and message.reference.message_id:
            reference_message = await message.channel.fetch_message(message.reference.message_id)
        
        conversation = await get_recent_conversation(message.channel,
                                                    min_messages=10,
                                                    max_messages=30, 
                                                    max_age_minutes=30, 
                                                    reference_message=reference_message)
        
        # Extract original message with bot mention removed
        extracted_message = message.content.replace(f"<@{bot.user.id}>", "").strip()
        
        response = await container.ai_client.generate_famous_person_response(
            conversation=conversation,
            person=famous_person,
            original_message=extracted_message
        )
        
        await message.reply(f"**{famous_person.title()} would say:**\n\n{response}")
    except Exception as e:
        print(f"Error generating famous person response: {e}")
        await message.reply(f"Sorry, I couldn't determine what {famous_person.title()} would say.")

async def get_recent_conversation(channel, min_messages=10, max_messages=30, max_age_minutes=30, reference_message=None):
    """
    Fetch recent messages from the channel to build conversation context.
    
    Args:
        channel: The Discord channel to get messages from
        min_messages: Minimum number of messages to retrieve, regardless of age
        max_messages: Maximum number of messages to retrieve
        max_age_minutes: Maximum age of messages in minutes from the reference point
        reference_message: If provided, use this as the starting point for the conversation
    """
    all_messages = []
    
    if reference_message:
        all_messages.append(reference_message)
        async for msg in channel.history(limit=max_messages, before=reference_message):
            all_messages.append(msg)
    else:
        async for msg in channel.history(limit=max_messages):
            all_messages.append(msg)
    
    if not all_messages:
        return []
    
    filtered_messages = []
    for msg in all_messages:
        if not msg.author.bot and not msg.content.startswith(f"<@{bot.user.id}>"):
            filtered_messages.append(msg)
    
    if not filtered_messages:
        return []
    
    reference_time = filtered_messages[0].created_at
    cutoff_time = reference_time - datetime.timedelta(minutes=max_age_minutes)
    
    conversation_messages = []
    min_messages_collected = 0
    
    for msg in filtered_messages:
        if min_messages_collected < min_messages:
            conversation_messages.append((msg.author.name, msg.content))
            min_messages_collected += 1
            continue
            
        if msg.created_at < cutoff_time:
            break
            
        conversation_messages.append((msg.author.name, msg.content))

    conversation_messages.reverse()
    return conversation_messages

async def is_joke(payload) -> bool:
    if payload.message_id in cache.joke_cache:
        return cache.joke_cache[payload.message_id]
    
    channel = await bot.fetch_channel(payload.channel_id)
    message = await channel.fetch_message(payload.message_id)
    
    # First check if this is a reply
    if not message.reference:
                return False
        
    # Get the source message that was replied to
    source_message = await channel.fetch_message(message.reference.message_id)
    
    is_joke_result = await container.ai_client.is_joke(
        source_message.content,
        message.content
    )
    
    # Cache the result
    cache.joke_cache[payload.message_id] = is_joke_result
    return is_joke_result

async def save_joke(payload):
    channel = await bot.fetch_channel(payload.channel_id)
    joke_message = await channel.fetch_message(payload.message_id)
    
    # Get the source message that was replied to
    source_message = await channel.fetch_message(joke_message.reference.message_id)
    
        # Detect languages with "unknown" fallback
    try:
        source_lang = detect(source_message.content)
    except LangDetectException:
        source_lang = 'unknown'
        
    try:
        joke_lang = detect(joke_message.content)
    except LangDetectException:
        joke_lang = 'unknown'
    
    # Calculate total reactions
    reaction_count = sum(reaction.count for reaction in joke_message.reactions)
    
    # Use container.store instead of store
    container.store.save(
        source_message_id=source_message.id,
        joke_message_id=joke_message.id,
        source_message_content=source_message.content,
        joke_message_content=joke_message.content,
        reaction_count=reaction_count,
        source_language=source_lang,
        joke_language=joke_lang
    )

async def retract_joke(payload):
    channel = await bot.fetch_channel(payload.channel_id)
    message = await channel.fetch_message(payload.message_id)

    if message.author.id == bot.user.id and await check_should_delete(message):
        await message.delete()

async def process_joke_request(payload, country=None):
    config = container.store.get_guild_config(payload.guild_id)
    
    # Skip country jokes if disabled
    if country and not config.enable_country_jokes:
        return

    channel = await bot.fetch_channel(payload.channel_id)
    message = await channel.fetch_message(payload.message_id)

    # Detect message language
    try:
        language = detect(message.content)
    except LangDetectException:
        language = 'en'  # Default to English if detection fails

    if country:
        joke = await container.joke_generator.generate_country_joke(message.content, country)
    else:
        joke = await container.joke_generator.generate_joke(message.content, language)
    
    # Send direct reply
    reply_message = await message.reply(joke)
    
    # Try to send to configured archive channel
    if config.archive_channel_id:
        try:
            archive_channel = await bot.fetch_channel(config.archive_channel_id)
            message_link = f"https://discord.com/channels/{payload.guild_id}/{payload.channel_id}/{payload.message_id}"
            archive_response = f"**Original message**: {message_link}\n{joke}"
            await archive_channel.send(archive_response)
        except Exception as e:
            print(f"Failed to send to archive channel: {e}")
    
    # Delete after timeout if configured
    if config.delete_jokes_after_minutes > 0:
        await asyncio.sleep(config.delete_jokes_after_minutes * 60)
        try:
            await reply_message.delete()
        except nextcord.errors.NotFound:
            pass

async def check_should_delete(message: nextcord.Message) -> bool:
    config = container.store.get_guild_config(message.guild.id)
    if config.downvote_reaction_threshold <= 0:
        return False

    upvotes = sum(r.count for r in message.reactions if str(r.emoji) == "👍")
    downvotes = sum(r.count for r in message.reactions if str(r.emoji) == "👎")
    
    return (downvotes - upvotes) >= config.downvote_reaction_threshold

# Get bot token from environment variable
TOKEN = os.getenv('DISCORD_TOKEN')
if not TOKEN:
    raise ValueError("Discord token not found in environment variables!")

bot.run(TOKEN)