import asyncio
from enum import Enum
from types import SimpleNamespace
import logging
from collections.abc import Awaitable, Callable

import nextcord

from opentelemetry.trace import SpanKind, Status, StatusCode

from container import container  # Import the instance instead of the class
from conversation_graph import ConversationGraphBuilder, ConversationMessage
from message_node import MessageNode


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
    def from_str(cls, value: str) -> "BotCommand | None":
        try:
            return next(cmd for cmd in cls if cmd.value == value.lower())
        except StopIteration:
            return None


@bot.event
async def on_ready() -> None:
    logger.info(f"Logged in as {bot.user}")
    # Initialize bot-dependent services
    container.user_resolver.set_bot_client(bot)


@bot.event
async def on_raw_reaction_add(payload: nextcord.RawReactionActionEvent):
    async with container.telemetry.async_create_span("on_raw_reaction_add", kind=SpanKind.CONSUMER) as span:
        span.set_attribute("guild_id", str(payload.guild_id))
        container.telemetry.increment_reaction_counter(payload)

        try:
            emoji_str = str(payload.emoji)
            country = await container.country_resolver.get_country_from_flag(emoji_str)
            message_key = (payload.message_id, emoji_str)

            is_clown = emoji_str == "🤡"
            is_country = country is not None
            is_thumbs_down = emoji_str == "👎"
            is_wisdom = emoji_str in ("🧔", "🧠") or payload.emoji.id == 1180114631574962196

            if (is_clown or is_country) and message_key not in cache.processed_messages:
                cache.processed_messages.add(message_key)
                await process_joke_request(payload, country)
            elif is_wisdom and message_key not in cache.processed_messages:
                cache.processed_messages.add(message_key)
                await process_wisdom_request(payload)
            elif is_thumbs_down:
                await retract_joke(payload)
            elif await is_joke(payload):
                await save_joke(payload)
        except ValueError as e:
            logger.error(f"Error processing reaction: {e}", exc_info=True)


@bot.event
async def on_message(message: nextcord.Message):
    timer = container.telemetry.metrics.timer()
    async with container.telemetry.async_create_span("on_message", kind=SpanKind.CONSUMER) as span:
        span.set_attribute("guild_id", str(message.guild.id))
        container.telemetry.increment_message_counter(message)

        # Ingest message into transient memory using materialized content
        materialized_message = await discord_to_message_node(message)
        await container.memory_manager.ingest_message(message.guild.id, materialized_message)

        if message.author.bot:
            return

        if not await should_reply(message):
            return

        # First check if this is a bot command
        is_command = await process_bot_commands(message)
        if is_command:
            return

        # Replace bot mention with "BOT" and pass the whole message to the router
        processed_message = message.content.replace(f"<@{bot.user.id}>", "BOT").strip()
        route, params = await container.ai_router.route_request(processed_message)

        reply = None
        if route == "FAMOUS":
            conversation_fetcher = create_conversation_fetcher(message)

            response = await container.famous_person_generator.handle_request(
                params, processed_message, conversation_fetcher, message.guild.id
            )
            reply = await message.reply(response)

        elif route == "GENERAL":
            conversation_fetcher = create_conversation_fetcher(message)

            response = await container.general_query_generator.handle_request(
                params, conversation_fetcher, message.guild.id, bot.user
            )
            if response is not None:
                reply = await message.reply(response)

        elif route == "FACT" and params:
            response = await container.fact_handler.handle_request(params, message.guild.id)
            reply = await message.reply(response)

        # Record reply latency once if a reply was sent
        if reply is not None:
            container.telemetry.metrics.message_latency.record(
                timer(), {"route": route, "guild_id": str(message.guild.id)}
            )


@bot.event
async def on_message_edit(before: nextcord.Message, after: nextcord.Message):
    async with container.telemetry.async_create_span("on_message_edit", kind=SpanKind.CONSUMER) as span:
        span.set_attribute("guild_id", str(after.guild.id))
        # Ignore edits from bots
        if after.author.bot:
            return

        # Ingest the updated message content, including embeds.
        # Caching in the materialization process handles efficiency.
        materialized_message = await discord_to_message_node(after)
        await container.memory_manager.ingest_message(after.guild.id, materialized_message)


async def should_reply(message: nextcord.Message) -> bool:
    """
    Check if the bot should reply to this message.
    Returns True if the message mentions the bot or is a reply to the bot.
    """
    # Check if bot is mentioned
    if f"<@{bot.user.id}>" in message.content:
        return True

    # Check if this is a reply to the bot
    if message.reference and message.reference.message_id:
        try:
            referenced_message = await message.channel.fetch_message(message.reference.message_id)
            return referenced_message.author.id == bot.user.id
        except Exception as e:
            logger.error(f"Error checking if message is reply to bot: {e}", exc_info=True)

    return False


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

    config = await container.store.get_guild_config(message.guild.id)

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
• Archive Channel: {f"<#{config.archive_channel_id}>" if config.archive_channel_id else "Disabled"}
• Auto-delete after: {config.delete_jokes_after_minutes} minutes (0 = never)
• Delete on downvotes: {config.downvote_reaction_threshold} (0 = disabled)
• Country jokes: {"Enabled" if config.enable_country_jokes else "Disabled"}
"""
        await message.reply(settings_text)
        return True

    if command == BotCommand.SET_ARCHIVE_CHANNEL:
        if not message.channel_mentions:
            config.archive_channel_id = 0  # Disable archiving
            await container.store.save_guild_config(config)
            await message.reply("Joke archiving has been disabled.")
            return True

        config.archive_channel_id = message.channel_mentions[0].id
        await container.store.save_guild_config(config)
        await message.reply(f"Jokes will now be archived in {message.channel_mentions[0].mention}")
        return True

    elif command == BotCommand.DELETE_JOKES_AFTER:
        try:
            minutes = int(args[1])
            if minutes < 0:
                raise ValueError
            config.delete_jokes_after_minutes = minutes
            await container.store.save_guild_config(config)
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
            await container.store.save_guild_config(config)
            await message.reply(f"Jokes will be deleted when downvotes - upvotes >= {threshold}")
        except (IndexError, ValueError):
            await message.reply("Please provide a valid threshold number!")
        return True

    elif command == BotCommand.ENABLE_COUNTRY_JOKES:
        try:
            enable = args[1].lower() == "true"
            config.enable_country_jokes = enable
            await container.store.save_guild_config(config)
            await message.reply(f"Country jokes {'enabled' if enable else 'disabled'}")
        except IndexError:
            await message.reply("Please specify true or false!")
        return True

    return False


async def discord_to_message_node(message: nextcord.Message) -> MessageNode:
    """Convert Discord message to MessageNode."""
    reference_id = None
    if message.reference and message.reference.message_id:
        reference_id = message.reference.message_id

    mentioned_user_ids = [user.id for user in message.mentions]

    content = message.content

    # Process all content (articles and images) through AttachmentProcessor
    if message.embeds or message.attachments:
        try:
            embeddings_xml = await container.attachment_processor.process_all_content(
                message.attachments, message.embeds
            )
            if embeddings_xml:
                content = f"{content} {embeddings_xml}" if content.strip() else embeddings_xml
        except Exception as e:
            logger.error(f"Error processing content for message {message.id}: {e}", exc_info=True)

    return MessageNode(
        id=message.id,
        content=content,
        author_id=message.author.id,
        channel_id=message.channel.id,
        mentioned_user_ids=mentioned_user_ids,
        created_at=message.created_at,
        reference_id=reference_id,
    )


async def get_recent_conversation(
    channel: nextcord.TextChannel,
    min_messages: int = 10,
    max_messages: int = 30,
    max_age_minutes: int = 30,
    reference_message: nextcord.Message | None = None,
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
    async def fetch_message(message_id: int) -> nextcord.Message | None:
        async with container.telemetry.async_create_span("fetch_message") as span:
            span.set_attribute("message_id", message_id)
            try:
                message = await channel.fetch_message(message_id)
                span.set_attribute("found", True)
                return message
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR))
                span.set_attribute("found", False)
                logger.error(f"Failed to fetch message {message_id}: {e}", exc_info=True)
                return None

    async def fetch_history(message_id: int | None) -> list[nextcord.Message]:
        async with container.telemetry.async_create_span("fetch_history") as span:
            span.set_attribute("message_id", message_id or "channel_start")

            message = None
            if message_id:
                try:
                    message = await channel.fetch_message(message_id)
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR))
                    logger.error(
                        f"Failed to fetch message {message_id} for history: {e}",
                        exc_info=True,
                    )
                    return []

            messages = []
            try:
                async for msg in channel.history(limit=100, before=message):
                    messages.append(msg)
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR))
                logger.error(f"Error fetching history: {e}", exc_info=True)

            span.set_attribute("messages_fetched", len(messages))
            return messages

    # Determine trigger message
    trigger_message = None
    if reference_message:
        trigger_message = reference_message
    else:
        # Get most recent message as trigger
        try:
            async for msg in channel.history(limit=1):
                trigger_message = msg
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
            telemetry=container.telemetry,
        )

        conversation = await builder.build_conversation_graph(
            trigger_message=trigger_message,
            min_linear=min_messages,
            max_total=max_messages,
            time_threshold_minutes=max_age_minutes,
            discord_to_message_node_func=discord_to_message_node,
        )

        span.set_attribute("total_messages", len(conversation))

    return conversation


def create_conversation_fetcher(
    message: nextcord.Message,
) -> Callable[[], Awaitable[list[ConversationMessage]]]:
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
            reference_message=reference_message,
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
        source_message.content, message.content, message_id=payload.message_id
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
        reaction_count=reaction_count,
    )


async def retract_joke(payload: nextcord.RawReactionActionEvent) -> None:
    channel = await bot.fetch_channel(payload.channel_id)
    message = await channel.fetch_message(payload.message_id)

    if message.author.id == bot.user.id and await check_should_delete(message):
        await message.delete()
        container.telemetry.metrics.message_deletions.add(1, {"reason": "downvotes", "guild_id": str(message.guild.id)})


async def process_joke_request(payload: nextcord.RawReactionActionEvent, country: str | None = None) -> None:
    config = await container.store.get_guild_config(payload.guild_id)

    # Skip country jokes if disabled
    if country and not config.enable_country_jokes:
        return

    channel = await bot.fetch_channel(payload.channel_id)
    message = await channel.fetch_message(payload.message_id)

    language = await container.language_detector.detect_language(message.content)

    if country:
        joke = await container.joke_generator.generate_country_joke(message.content, country)
    else:
        joke = await container.joke_generator.generate_joke(message.content, language)

    # Send direct reply
    reply_message = await message.reply(joke)
    # Count jokes generated
    attrs = {"language": language or "unknown", "guild_id": str(payload.guild_id)}
    if country:
        attrs["country"] = country
    container.telemetry.metrics.jokes_generated.add(1, attrs)

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


async def process_wisdom_request(payload: nextcord.RawReactionActionEvent) -> None:
    """Handle wisdom generation request triggered by beard or brain emoji."""
    config = await container.store.get_guild_config(payload.guild_id)

    channel = await bot.fetch_channel(payload.channel_id)
    message = await channel.fetch_message(payload.message_id)

    async def fetch_conversation():
        return await get_recent_conversation(
            channel,
            min_messages=10,
            max_messages=30,
            max_age_minutes=30,
            reference_message=message,
        )

    wisdom = await container.wisdom_generator.generate_wisdom(
        trigger_message_content=message.content,
        conversation_fetcher=fetch_conversation,
        guild_id=payload.guild_id,
    )

    if wisdom is None:
        return

    reply_message = await message.reply(wisdom)

    container.telemetry.metrics.wisdom_generated.add(1, {"guild_id": str(payload.guild_id)})

    # Try to send to configured archive channel
    if config.archive_channel_id:
        try:
            archive_channel = await bot.fetch_channel(config.archive_channel_id)
            message_link = f"https://discord.com/channels/{payload.guild_id}/{payload.channel_id}/{payload.message_id}"
            archive_response = f"**Original message**: {message_link}\n{wisdom}"
            await archive_channel.send(archive_response)
        except Exception as e:
            logger.error(f"Failed to send to archive channel: {e}", exc_info=True)

    if config.delete_jokes_after_minutes > 0:
        asyncio.create_task(delete_message_later(reply_message, config.delete_jokes_after_minutes * 60))


async def delete_message_later(message: nextcord.Message, delay_seconds: int) -> None:
    """Delete a message after a delay without blocking the caller."""
    await asyncio.sleep(delay_seconds)
    try:
        await message.delete()
        container.telemetry.metrics.message_deletions.add(1, {"reason": "timeout", "guild_id": str(message.guild.id)})
    except nextcord.errors.NotFound:
        # Message might have been deleted already
        pass


async def check_should_delete(message: nextcord.Message) -> bool:
    config = await container.store.get_guild_config(message.guild.id)
    if config.downvote_reaction_threshold <= 0:
        return False

    upvotes = sum(r.count for r in message.reactions if str(r.emoji) == "👍")
    downvotes = sum(r.count for r in message.reactions if str(r.emoji) == "👎")

    return (downvotes - upvotes) >= config.downvote_reaction_threshold


# Get bot token from centralized configuration
bot.run(container.config.discord_token)
