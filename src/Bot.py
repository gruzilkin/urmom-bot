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
    if str(payload.emoji) != "🤡":
        return

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