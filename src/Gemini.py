import re
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get API key, temperature and model name from environment
api_key = os.getenv('GEMINI_API_KEY')
temperature = float(os.getenv('GEMINI_TEMPERATURE', 1.2))
model_name = os.getenv('GEMINI_MODEL')

if not api_key:
    raise ValueError("Gemini API key not found in environment variables!")
if not model_name:
    raise ValueError("Gemini model name not found in environment variables!")

genai.configure(api_key=api_key)

def generate_joke(content):
    # Create the model
    generation_config = {
        "temperature": temperature,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
        model_name=model_name,
        generation_config=generation_config,
    )

    # it makes sense to collect ur mom jokes from chats along with reactions to create a dataset for model tuning or for sending alongside as prompt

    prompt = [
        """You are a chatbot that receives a message and should generate a joke.
        One of the jokes is ur mom joke, ur mom joke follows the pattern of replacing the subject or the object in a phrase with \"ur mom\" without adding much extra details.
        Make it as lewd and preposterous as possible, carefully replace the subject and/or some objects in order to achieve the most outrageous result.
        The second type is \"In Soviet Russia\" joke. \"In Soviet Russia\", also called the Russian reversal, is a joke template taking the general form \"In America you do X to/with Y; in Soviet Russia Y does X to/with you\".
        Typically the American clause describes a harmless ordinary activity and the inverted Soviet form something menacing or dysfunctional, satirizing life under communist rule, or in the \"old country\".
        Sometimes the first clause is omitted, and sometimes either clause or both are deliberately rendered with English grammatical errors stereotypical of Russians.
        You can combine the two patterns if it results in a funny joke.
        Make sure that the joke is grammatically correct, check for subject-verb agreement, update pronouns after replacing subjects and objects.
        If the result looks absurd and funny then it's a good joke, you should surround your response with *joke* tag before the joke and after so that I could automatically extract it with a script.
        Respond with *none* if the sentence is not a good material for such a joke or if it doesn't sound funny in the end.
        """,
        "input: My son is bioterrorising me. He walked in my office, sat down for a few minutes, farted, and then immediately left",
        "output: *joke*Ur mom is bioterrorising me. She walked in my office, sat down for a few minutes, farted, and then immediately left*joke*",
        "input: So Trump outlawed LGBT and there's no protests whatsoever, it's as if it was never a big movement and in reality people don't support it unless forced to",
        "output: *joke*So Trump outlawed ur mom and there's no protests whatsoever, it's as if it was never a big movement and in reality people don't support her unless forced to*joke*",
        "input: Jim likes cake.",
        "output: *joke*Jim likes ur mom.*joke*",
        "input: tldr, its basically switch 1 with some extra coloured plastic",
        "output: *joke*tldr, its basically ur mom with some extra coloured plastic*joke*",
        "input: Phones used to be 500$ now it's 1500$",
        "output: *joke*Ur mom used to be 500$ now it's 1500$*joke*",
        "input: Everyone knows that a tablet that can't do FaceTime is dead on arrival",
        "output: *joke*Everyone knows that ur mom can do FaceTime*joke*",
        "input: Ouch. You been putting strange objects in your ear?",
        "output: *joke*ur mom has been putting strange objects in her ear*joke*",
        "input: Apparently it's a trophy for getting all achievements",
        "output: *joke*It's a trophy for getting ur mom*joke*",
        "input: He's harder in multiplayer",
        "output: *joke*Ur mom's harder in multiplayer*joke*",
        "input: any of you on globalsign's slack ?",
        "output: *joke*I'm on ur mom atm*joke*",
        "input: just wait till you see what's on the roadmap",
        "output: *joke*Ur mom's on the roadmap*joke*",
        "input: i doubt that would fit on 4TB",
        "output: *joke*i doubt ur mom would fit on 4TB*joke*",
        "input: is it possible you aren't connected to their network somehow? like with a firewall blocking it",
        "output: *joke*Ur mom's blocking it*joke*",
        "input: I use USB-C to DP cable",
        "output: *joke*I use USB-C to DP ur mom*joke*",
        "input: yeah.. and I think our standards get lower than they used to be, thx to netflix series",
        "output: *joke*Ur mom's standards get lower than they used to be*joke*",
        "input: How was it down there?",
        "output: *joke*How was ur mom down there*joke*",
        "input: Nissin has good cheese cake, it's in the freezer section",
        "output: *joke*ur mom's in the freezer section*joke*",
        "input: fun story: As I was taking the garbage downstairs last night (2am), there was a huge (female I guess?) mosquito in the elevator. It was flying around, I really didn't want it to land on me, so I whacked it with the garbage bag, it died. I walk out of the elevator (and I make note of where it died), drop off the garbage bag (this takes 20 seconds), and go back in the elevator.....I check the mosquito......it's gone, no where to be seen........must be a big spider/something living in our elevator.",
        "output: *joke*ur mom got whacked at 2am last night*joke*",
        "input: Musk is working on it",
        "output: *joke*Musk is working on ur mom*joke*",
        "input: sound via bone conduction",
        "output: *joke*I typically use ur mom for sound via bone conduction*joke*",
        "input: this website is too much fun",
        "output: *joke*ur mom is too much fun*joke*",
        "input: Yes, everything can be done with a couple of command lines",
        "output: *joke*ur mom can be done with a couple of command lines*joke*",
        "input: that is a predicament. Maybe your charting library supports doing it somehow though?",
        "output: *joke*that is a predicament. Maybe your charting library supports doing ur mom somehow though?*joke*",
        "input: you just reminded me I need to clean ours.  hasn't been cleaned in like 8 months",
        "output: *joke*ur mom hasn't been cleaned in like 8 months*joke*",
        "input: I was just reading, they think it's falling because US interest rates are going up and Japan's are ultra-low",
        "output: *joke*ur mom's interest rate is going down*joke*",
        "input: what is that blue ball at the base of it?",
        "output: *joke*prolly spent the night with ur mom*joke*",
        "input: I'm seeing Top Gun on Sunday!",
        "output: *joke*Ur mom's seeing top gun on Sunday, whatever that might mean*joke*",
        "input: is there anything you own that you would NOT recommend to anyone ?",
        "output: *joke*yes, ur mom*joke*",
        "input: Ooo, so you need the sausages asap",
        "output: *joke*ur mom need the sausages asap*joke*",
        "input: yes , I ended up using the biggest size",
        "output: *joke*ur mom ended up using the biggest size*joke*",
        "input: that would be some weak ass theory",
        "output: *joke*ur mom's a weak ass theory*joke*",
        "input: whats leaking?",
        "output: *joke*ur mom*joke*",
        "input: And I tried touching midnight color at a store - my hands were just leaving wet marks on it",
        "output: *joke*ur mom leaves wet marks*joke*",
        "input: Markets are having a good rally recently",
        "output: *joke*ur mom's having a good rally*joke*",
        "input: Looks like a Boston dynamics robot",
        "output: *joke*ur mom looks like a Boston dynamics robot*joke*",
        "input: best show I've seen in a while",
        "output: *joke*ur mom's the best show I've seen in a while*joke*",
        "input: GCC isn't exactly a day job",
        "output: *joke*ur mom isn't exactly a day job*joke*",
        "input: do you guys have the ability to pin messages?",
        "output: *joke*I have the ability to pin ur mom*joke*",
    ]

    response = model.generate_content(prompt + ["input: " + content, "output: "])

    print (f"source: {content}")
    print (f"response: {response.text}")

    joke_match = re.search(r'\*joke\*(.*?)\*joke\*', response.text, re.DOTALL)
    if joke_match:
        joke = joke_match.group(1).strip()
        return joke
    else:
        return "I need some better urmom joke material to work with"