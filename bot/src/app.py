import nextcord
from dotenv import load_dotenv
from container import container  # Import the instance instead of the class
import os
import asyncio

load_dotenv()

BOT_CHANNEL_NAME = os.getenv('BOT_CHANNEL_NAME')
if not BOT_CHANNEL_NAME:
    raise ValueError("Bot channel name not found in environment variables!")

intents = nextcord.Intents.default()
intents.message_content = True  # MUST have this to receive message content

bot = nextcord.Client(intents=intents)

# Set to store IDs of messages we've already responded to
processed_messages = set()

guild_channel_mapping = {}

# Get timeout value in seconds from env, default to 600 (10 minutes)
DELETE_JOKES_AFTER = os.getenv('DELETE_JOKES_AFTER')
if not DELETE_JOKES_AFTER:
    raise ValueError("DELETE_JOKES_AFTER not found in environment variables!")
try:
    DELETE_JOKES_AFTER = int(DELETE_JOKES_AFTER)
except ValueError:
    raise ValueError("DELETE_JOKES_AFTER must be a valid integer!")

async def get_bot_channel_id(guild_id: int) -> int:
    """Get or find bot channel ID for a specific guild"""
    if guild_id in guild_channel_mapping:
        return guild_channel_mapping[guild_id]

    # Find the guild by ID
    guild = bot.get_guild(guild_id)
    if not guild:
        raise ValueError(f"Could not find guild {guild_id}")

    # Find the bot channel in this guild
    channel = nextcord.utils.get(guild.text_channels, name=BOT_CHANNEL_NAME)
    if not channel:
        raise ValueError(f"Could not find channel #{BOT_CHANNEL_NAME} in guild {guild.name}")

    # Cache the channel ID
    guild_channel_mapping[guild_id] = channel.id
    print(f'Found bot channel in {guild.name}: #{channel.name} ({channel.id})')
    return channel.id

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

        if (is_clown or is_country) and message_key not in processed_messages:
            await process_joke_request(payload, country)
            processed_messages.add(message_key)
        
        if await is_joke(payload):
            await save_joke(payload)
    except ValueError as e:
        print(f"Error processing reaction: {e}")

async def is_joke(payload):
    channel = await bot.fetch_channel(payload.channel_id)
    message = await channel.fetch_message(payload.message_id)
    
    # First check if this is a reply
    if not message.reference:
        return False
        
    content = message.content.lower()
    return "ur" in content and "mom" in content


async def save_joke(payload):
    channel = await bot.fetch_channel(payload.channel_id)
    joke_message = await channel.fetch_message(payload.message_id)
    
    # Get the source message that was replied to
    source_message = await channel.fetch_message(joke_message.reference.message_id)
    
    # Calculate total reactions
    reaction_count = sum(reaction.count for reaction in joke_message.reactions)
    
    # Use container.store instead of store
    container.store.save(
        source_message_id=source_message.id,
        joke_message_id=joke_message.id,
        source_message_content=source_message.content,
        joke_message_content=joke_message.content,
        reaction_count=reaction_count
    )

async def process_joke_request(payload, country=None):
    channel = await bot.fetch_channel(payload.channel_id)
    message = await channel.fetch_message(payload.message_id)
    bot_channel_id = await get_bot_channel_id(payload.guild_id)
    bot_channel = await bot.fetch_channel(bot_channel_id)

    if country:
        joke = await container.joke_generator.generate_country_joke(message.content, country)
    else:
        joke = await container.joke_generator.generate_joke(message.content)
    
    # Send direct reply with just the joke
    reply_message = await message.reply(joke)
    
    # Send full message with link to bot channel
    message_link = f"https://discord.com/channels/{payload.guild_id}/{payload.channel_id}/{payload.message_id}"
    bot_channel_response = f"**Original message**: {message_link}\n{joke}"
    await bot_channel.send(bot_channel_response)
    
    await asyncio.sleep(DELETE_JOKES_AFTER)
    try:
        await reply_message.delete()
    except nextcord.errors.NotFound:
        pass

# Get bot token from environment variable
TOKEN = os.getenv('DISCORD_TOKEN')
if not TOKEN:
    raise ValueError("Discord token not found in environment variables!")

bot.run(TOKEN)