import asyncio
import datetime
import os
from enum import Enum
from types import SimpleNamespace
import logging
from typing import Callable, Awaitable
from functools import lru_cache

import nextcord
from dotenv import load_dotenv
from langdetect import detect, LangDetectException
from goose3 import Goose

from opentelemetry.trace import SpanKind, Status, StatusCode

from container import container  # Import the instance instead of the class
from conversation_graph import ConversationGraphBuilder, MessageNode, ConversationMessage

load_dotenv()

intents = nextcord.Intents.default()
intents.message_content = True  # MUST have this to receive message content

bot = nextcord.Client(intents=intents)

# Create cache object to hold bot state
cache = SimpleNamespace()
cache.processed_messages = set()

# Set up a module-level logger
logger = logging.getLogger(__name__)

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
async def on_ready() -> None:
    logger.info(f'Logged in as {bot.user}')
    # Initialize bot-dependent services
    container.user_resolver.set_bot_client(bot)

@bot.event
async def on_raw_reaction_add(payload: nextcord.RawReactionActionEvent):
    async with container.telemetry.async_create_span("on_raw_reaction_add", kind=SpanKind.CONSUMER) as span:
        container.telemetry.increment_reaction_counter(payload)
        
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
            logger.error(f"Error processing reaction: {e}", exc_info=True)

@bot.event
async def on_message(message: nextcord.Message):
    async with container.telemetry.async_create_span("on_message", kind=SpanKind.CONSUMER) as span:
        container.telemetry.increment_message_counter(message)
        
        if message.author.bot:
            return
        
        if f"<@{bot.user.id}>" not in message.content:
            return

        # First check if this is a bot command
        is_command = await process_bot_commands(message)
        if is_command:
            return

        # Replace bot mention with "BOT" and pass the whole message to the router
        processed_message = message.content.replace(f"<@{bot.user.id}>", "BOT").strip()
        router_decision = await container.ai_router.route_request(processed_message)
        
        if router_decision.route == "FAMOUS":
            conversation_fetcher = create_conversation_fetcher(message)
            
            response = await container.famous_person_generator.handle_request(
                router_decision.famous_params, processed_message, conversation_fetcher, message.guild.id
            )
            await message.reply(response)
        
        elif router_decision.route == "GENERAL":
            conversation_fetcher = create_conversation_fetcher(message)
            
            response = await container.general_query_generator.handle_request(
                router_decision.general_params, conversation_fetcher, message.guild.id
            )
            await message.reply(response)

        elif router_decision.route == "FACT" and router_decision.fact_params:
            response = await container.fact_handler.handle_request(
                router_decision.fact_params, message.guild.id
            )
            await message.reply(response)

async def process_bot_commands(message: nextcord.Message) -> bool:
    """
    Process bot commands from the message if any.
    Returns True if a command was processed, False otherwise.
    """
    # Extract command arguments
    args = message.content.split()[1:]
    if not args:
        return False

    command = BotCommand.from_str(args[0])
    if not command:
        return False
        
    # Check if user is admin before processing commands
    if not message.author.guild_permissions.administrator:
        await message.reply("Sorry, only administrators can use bot commands!")
        return True
            
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
        return True

    if command == BotCommand.SETTINGS:
        settings_text = f"""
Current settings:
• Archive Channel: {f'<#{config.archive_channel_id}>' if config.archive_channel_id else 'Disabled'}
• Auto-delete after: {config.delete_jokes_after_minutes} minutes (0 = never)
• Delete on downvotes: {config.downvote_reaction_threshold} (0 = disabled)
• Country jokes: {'Enabled' if config.enable_country_jokes else 'Disabled'}
"""
        await message.reply(settings_text)
        return True

    if command == BotCommand.SET_ARCHIVE_CHANNEL:
        if not message.channel_mentions:
            config.archive_channel_id = 0  # Disable archiving
            container.store.save_guild_config(config)
            await message.reply("Joke archiving has been disabled.")
            return True
            
        config.archive_channel_id = message.channel_mentions[0].id
        container.store.save_guild_config(config)
        await message.reply(f"Jokes will now be archived in {message.channel_mentions[0].mention}")
        return True

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
        return True

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
        return True

    elif command == BotCommand.ENABLE_COUNTRY_JOKES:
        try:
            enable = args[1].lower() == "true"
            config.enable_country_jokes = enable
            container.store.save_guild_config(config)
            await message.reply(f"Country jokes {'enabled' if enable else 'disabled'}")
        except IndexError:
            await message.reply("Please specify true or false!")
        return True
        
    return False


def discord_to_message_node(message: nextcord.Message) -> MessageNode:
    """Convert Discord message to MessageNode."""
    reference_id = None
    if message.reference and message.reference.message_id:
        reference_id = message.reference.message_id
    
    # Extract mentioned user IDs
    mentioned_user_ids = [user.id for user in message.mentions]
    
    return MessageNode(
        id=message.id,
        content=message.content,
        author_id=message.author.id,
        mentioned_user_ids=mentioned_user_ids,
        created_at=message.created_at,
        reference_id=reference_id,
        embeds=message.embeds
    )


@lru_cache(maxsize=128)
def extract_article(url: str) -> str:
    """Extract article content with LRU caching."""
    with container.telemetry.create_span("extract_article") as span:
        span.set_attribute("url", url)
        try:
            article = Goose().extract(url)
            content = article.cleaned_text if article.cleaned_text else ""
            span.set_attribute("content_length", len(content))
            if content:
                logger.info(f"Extracted article content from {url}: {content[:500]}{'...' if len(content) > 500 else ''}")
            else:
                logger.info(f"No content extracted from {url}")
            return content
        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR))
            logger.error(f"Error extracting article from {url}: {e}", exc_info=True)
            return ""



async def get_recent_conversation(
    channel: nextcord.TextChannel,
    min_messages: int = 10,
    max_messages: int = 30,
    max_age_minutes: int = 30,
    reference_message: nextcord.Message | None = None
) -> list[ConversationMessage]:
    """
    Fetch recent conversation using graph-based clustering.
    
    Args:
        channel: The Discord channel to get messages from
        min_messages: Minimum number of messages to retrieve (now min_linear)
        max_messages: Maximum number of messages to retrieve (now max_total)
        max_age_minutes: Time threshold for temporal connections (now time_threshold_minutes)
        reference_message: If provided, use this as the starting point for the conversation
        
    Returns:
        List of ConversationMessage objects with timestamps in chronological order
    """
    # Create Discord API adapter functions
    async def fetch_message(message_id: int) -> MessageNode | None:
        async with container.telemetry.async_create_span("fetch_message") as span:
            span.set_attribute("message_id", message_id)
            try:
                message = await channel.fetch_message(message_id)
                result = discord_to_message_node(message)
                span.set_attribute("found", True)
                return result
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR))
                span.set_attribute("found", False)
                logger.error(f"Failed to fetch message {message_id}: {e}", exc_info=True)
                return None
    
    async def fetch_history(message_id: int | None) -> list[MessageNode]:
        async with container.telemetry.async_create_span("fetch_history") as span:
            span.set_attribute("message_id", message_id or "channel_start")
            
            message = None
            if message_id:
                try:
                    message = await channel.fetch_message(message_id)
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR))
                    logger.error(f"Failed to fetch message {message_id} for history: {e}", exc_info=True)
                    return []
            
            messages = []
            try:
                async for msg in channel.history(limit=100, before=message):
                    messages.append(discord_to_message_node(msg))
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR))
                logger.error(f"Error fetching history: {e}", exc_info=True)
            
            span.set_attribute("messages_fetched", len(messages))
            return messages
    
    # Determine trigger message
    trigger_message = None
    if reference_message:
        trigger_message = discord_to_message_node(reference_message)
    else:
        # Get most recent message as trigger
        try:
            async for msg in channel.history(limit=1):
                trigger_message = discord_to_message_node(msg)
                break
        except Exception as e:
            logger.error(f"Failed to fetch most recent message from channel: {e}", exc_info=True)
            return []
    
    if not trigger_message:
        return []
    
    # Build conversation graph with telemetry
    with container.telemetry.create_span("build_conversation_graph") as span:
        builder = ConversationGraphBuilder(
            fetch_message=fetch_message,
            fetch_history=fetch_history,
            telemetry=container.telemetry
        )
        
        conversation = await builder.build_conversation_graph(
            trigger_message=trigger_message,
            min_linear=min_messages,
            max_total=max_messages,
            time_threshold_minutes=max_age_minutes,
            article_extractor=extract_article
        )
        
        span.set_attribute("total_messages", len(conversation))
    
    return conversation

def create_conversation_fetcher(message: nextcord.Message) -> Callable[[], Awaitable[list[ConversationMessage]]]:
    """
    Create a parameterless lambda that encapsulates conversation fetching logic.
    
    Args:
        message: The Discord message to use for conversation context
        
    Returns:
        async function: Parameterless function that returns conversation history
    """
    async def fetch_conversation():
        reference_message = None
        if message.reference and message.reference.message_id:
            reference_message = await message.channel.fetch_message(message.reference.message_id)
        
        return await get_recent_conversation(
            message.channel,
            min_messages=10,
            max_messages=30, 
            max_age_minutes=30, 
            reference_message=reference_message
        )
    
    return fetch_conversation

async def is_joke(payload: nextcord.RawReactionActionEvent) -> bool:
    channel = await bot.fetch_channel(payload.channel_id)
    message = await channel.fetch_message(payload.message_id)
    
    # First check if this is a reply
    if not message.reference:
        return False
        
    # Get the source message that was replied to
    source_message = await channel.fetch_message(message.reference.message_id)
    
    # Use JokeGenerator's is_joke method with caching
    return await container.joke_generator.is_joke(
        source_message.content,
        message.content,
        message_id=payload.message_id
    )

async def save_joke(payload: nextcord.RawReactionActionEvent) -> None:
    channel = await bot.fetch_channel(payload.channel_id)
    joke_message = await channel.fetch_message(payload.message_id)
    
    # Get the source message that was replied to
    source_message = await channel.fetch_message(joke_message.reference.message_id)
    
    # Calculate total reactions
    reaction_count = sum(reaction.count for reaction in joke_message.reactions)
    
    # Use JokeGenerator's save method
    await container.joke_generator.save_joke(
        source_message_id=source_message.id,
        source_message_content=source_message.content,
        joke_message_id=joke_message.id,
        joke_message_content=joke_message.content,
        reaction_count=reaction_count
    )

async def retract_joke(payload: nextcord.RawReactionActionEvent) -> None:
    channel = await bot.fetch_channel(payload.channel_id)
    message = await channel.fetch_message(payload.message_id)

    if message.author.id == bot.user.id and await check_should_delete(message):
        await message.delete()

async def process_joke_request(payload: nextcord.RawReactionActionEvent, country: str | None = None) -> None:
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
            logger.error(f"Failed to send to archive channel: {e}", exc_info=True)
    
    # Delete after timeout if configured
    if config.delete_jokes_after_minutes > 0:
        asyncio.create_task(delete_message_later(reply_message, config.delete_jokes_after_minutes * 60))

async def delete_message_later(message: nextcord.Message, delay_seconds: int) -> None:
    """Delete a message after a delay without blocking the caller."""
    await asyncio.sleep(delay_seconds)
    try:
        await message.delete()
    except nextcord.errors.NotFound:
        # Message might have been deleted already
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