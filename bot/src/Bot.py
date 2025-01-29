import nextcord
from dotenv import load_dotenv
from nextcord.ext import commands
from Gemini import generate_joke
import os

load_dotenv()

intents = nextcord.Intents.default()
intents.message_content = True  # MUST have this to receive message content

bot = commands.Bot(command_prefix="!", intents=intents)

# Set to store IDs of messages we've already responded to
processed_messages = set()

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user}')

@bot.event
async def on_raw_reaction_add(payload):
    # Only process 🤡 emoji reactions
    if str(payload.emoji) == "🤡":
        await process_joke_request(payload)
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
    
    # Create joke data structure with message contents
    joke_data = {
        "source_message_id": source_message.id,
        "joke_message_id": joke_message.id,
        "source_message_content": source_message.content,
        "joke_message_content": joke_message.content,
        "reaction_count": reaction_count
    }

    print(f"Saving joke: {joke_data}")

async def process_joke_request(payload):
    # Skip if already processed this message
    if payload.message_id in processed_messages:
        return

    channel = await bot.fetch_channel(payload.channel_id)
    message = await channel.fetch_message(payload.message_id)

    # Generate and send joke
    joke = generate_joke(message.content)
    await message.reply(joke)
    
    # Mark message as processed
    processed_messages.add(payload.message_id)


# Get bot token from environment variable
TOKEN = os.getenv('DISCORD_TOKEN')
if not TOKEN:
    raise ValueError("Discord token not found in environment variables!")

bot.run(TOKEN)