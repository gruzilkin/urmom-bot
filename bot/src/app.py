import nextcord
from dotenv import load_dotenv
from container import container  # Import the instance instead of the class
from utils import get_country_from_flag
import os

load_dotenv()

intents = nextcord.Intents.default()
intents.message_content = True  # MUST have this to receive message content

bot = nextcord.Client(intents=intents)

# Set to store IDs of messages we've already responded to
processed_messages = set()

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user}')

@bot.event
async def on_raw_reaction_add(payload):
    emoji_str = str(payload.emoji)
    country = get_country_from_flag(emoji_str)

    if emoji_str == "🤡":
        await process_joke_request(payload)
    elif country is not None:
        await process_country_joke_request(payload, country)
    elif await is_joke(payload):
        await save_joke(payload)

async def is_joke(payload):
    channel = await bot.fetch_channel(payload.channel_id)
    message = await channel.fetch_message(payload.message_id)
    
    # First check if this is a reply
    if not message.reference:
        return False
        
    content = message.content.lower()
    
    is_ur_mom_joke = "urmom" in content or "ur mom" in content
    is_soviet_russia_joke = "soviet" in content and "russia" in content

    return is_ur_mom_joke or is_soviet_russia_joke

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

async def process_joke_request(payload):
    # Skip if already processed this message
    if payload.message_id in processed_messages:
        return

    channel = await bot.fetch_channel(payload.channel_id)
    message = await channel.fetch_message(payload.message_id)

    sample_count = int(os.getenv('SAMPLE_JOKES_COUNT', '10'))
    # Use container.store and container.joke_generator
    joke = container.joke_generator.generate_joke(message.content, container.store.get_random_jokes(sample_count))
    await message.reply(joke)
    
    # Mark message as processed
    processed_messages.add(payload.message_id)

async def process_country_joke_request(payload, country):
    channel = await bot.fetch_channel(payload.channel_id)
    message = await channel.fetch_message(payload.message_id)

    # Use container.joke_generator
    joke = container.joke_generator.generate_country_joke(message.content, country)
    await message.reply(joke)

# Get bot token from environment variable
TOKEN = os.getenv('DISCORD_TOKEN')
if not TOKEN:
    raise ValueError("Discord token not found in environment variables!")

bot.run(TOKEN)