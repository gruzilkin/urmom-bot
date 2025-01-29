import re
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get API key from environment
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    raise ValueError("Gemini API key not found in environment variables!")

genai.configure(api_key=api_key)

def generate_joke(content):
    # Create the model
    generation_config = {
        "temperature": 1.2,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
        model_name="gemini-exp-1206",
        generation_config=generation_config,
    )

    prompt = [
        "You are a chatbot that receives a message and should generate a joke.One of the jokes is ur mom joke, ur mom joke follows the pattern of replacing the subject or the object in a phrase with \"ur mom\" without adding much extra details. Make it as lewd and preposterous as possible, carefully replace the subject and/or some objects in order to achieve the most outrageous result.\nThe second type is \"In Soviet Russia\" joke. \"In Soviet Russia\", also called the Russian reversal, is a joke template taking the general form \"In America you do X to/with Y; in Soviet Russia Y does X to/with you\". Typically the American clause describes a harmless ordinary activity and the inverted Soviet form something menacing or dysfunctional, satirizing life under communist rule, or in the \"old country\". Sometimes the first clause is omitted, and sometimes either clause or both are deliberately rendered with English grammatical errors stereotypical of Russians.\nYou can combine the two patterns if it results in a funny joke.\nMake sure that the joke is grammatically correct, check for subject-verb agreement, update pronouns after replacing subjects and objects. If the result looks absurd and funny then it's a good joke, you should surround your response with *joke* tag before the joke and after so that I could automatically extract it with a script.\nRespond with *none* if the sentence is not a good material for such a joke or if it doesn't sound funny in the end.",
        "input: My son is bioterrorising me. He walked in my office, sat down for a few minutes, farted, and then immediately left",
        "output: Ur mom is bioterrorising me. She walked in my office, sat down for a few minutes, farted, and then immediately left",
        "input: So Trump outlawed LGBT and there's no protests whatsoever, it's as if it was never a big movement and in reality people don't support it unless forced to",
        "output: So Trump outlawed ur mom and there's no protests whatsoever, it's as if it was never a big movement and in reality people don't support her unless forced to",
        "input: Jim like cake.",
        "output: Jim like ur mom.",
        "input: tldr, its basically switch 1 with some extra coloured plastic",
        "output: tldr, its basically ur mom with some extra coloured plastic",
        "input: Phones used to be 500$ now it's 1500$",
        "output: Ur mom used to be 500$ now it's 1500$",
        "input: Everyone knows that a tablet that can't do FaceTime is dead on arrival",
        "output: Everyone knows that ur mom can do FaceTime"]

    response = model.generate_content(prompt + ["input: " + content, "output: "])

    joke_match = re.search(r'\*joke\*(.*?)\*joke\*', response.text, re.DOTALL)
    if joke_match:
        joke = joke_match.group(1).strip()
        return joke
    else:
        return "I need some better urmom joke material to work with"