import nextcord
from dotenv import load_dotenv
from nextcord.ext import commands
from Gemini import generate_joke
import os

load_dotenv()

intents = nextcord.Intents.default()
intents.message_content = True  # MUST have this to receive message content

bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user}')

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    # Check if the bot is mentioned (tagged)
    if bot.user.mentioned_in(message):
        referenced_message = None
        if message.reference:
            referenced_message = await message.channel.fetch_message(message.reference.message_id)

        if referenced_message:
            # Generate the joke based on the original message
            joke = generate_joke(referenced_message.content)
            print(f"original message: {referenced_message.content}\njoke: {joke}")
            # Send the joke as a reply
            await message.reply(joke)


# Get bot token from environment variable
TOKEN = os.getenv('DISCORD_TOKEN')
if not TOKEN:
    raise ValueError("Discord token not found in environment variables!")

bot.run(TOKEN)