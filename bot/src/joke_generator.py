import logging
from collections.abc import Callable, Awaitable

from ai_client import AIClient
from conversation_graph import ConversationMessage
from open_telemetry import Telemetry
from store import Store
from language_detector import LanguageDetector
from memory_manager import MemoryManager
from conversation_formatter import ConversationFormatter
from schemas import YesNo
from opentelemetry.trace import SpanKind


logger = logging.getLogger(__name__)


class JokeGenerator:
    def __init__(
        self,
        joke_writer_client: AIClient,
        joke_classifier_client: AIClient,
        store: Store,
        telemetry: Telemetry,
        language_detector: LanguageDetector,
        conversation_formatter: ConversationFormatter,
        memory_manager: MemoryManager,
        sample_count: int = 10,
    ):
        self._joke_writer_client = joke_writer_client
        self._joke_classifier_client = joke_classifier_client
        self.store = store
        self.sample_count = sample_count
        self.telemetry = telemetry
        self.language_detector = language_detector
        self._conversation_formatter = conversation_formatter
        self._memory_manager = memory_manager
        self._joke_cache: dict[int, bool] = {}  # message_id -> bool cache

    def _extract_unique_user_ids(self, conversation: list[ConversationMessage]) -> set[int]:
        user_ids = set()
        for msg in conversation:
            user_ids.add(msg.author_id)
            user_ids.update(msg.mentioned_user_ids)
        return user_ids

    async def generate_joke(
        self,
        content: str,
        language: str,
        conversation_fetcher: Callable[[], Awaitable[list[ConversationMessage]]],
        guild_id: int,
    ) -> str:
        # Convert language code to language name
        language_name = await self.language_detector.get_language_name(language)

        sample_jokes = await self.store.get_random_jokes(self.sample_count)

        # Format sample jokes as XML examples
        examples_xml = ""
        if sample_jokes:
            examples_xml = "<examples>"
            for message, joke in sample_jokes:
                examples_xml += f"<example><message>{message}</message><joke>{joke}</joke></example>"
            examples_xml += "</examples>"

        conversation = await conversation_fetcher()
        conversation_block = await self._conversation_formatter.format_to_xml(guild_id, conversation)
        user_ids = self._extract_unique_user_ids(conversation)
        memories_block = await self._memory_manager.build_memory_prompt(guild_id, user_ids)

        # Create the prompt using format string
        russian_note = " In Russian, use the slang form 'твоя мамка'." if language == "ru" else ""
        prompt = f"""You are a chatbot that generates jokes in response to messages.
Read the message, conversation context, and any user memories,
then pick whichever joke format below produces the funniest
result. The format order is not a priority — always go with
the joke that lands hardest. The bot's identity is "ur mom"
jokes, so prefer that angle when the quality is comparable,
but never force a weak "ur mom" joke over a great alternative.
Freestyle is a last resort for when nothing else fits.

<formats>
<format name="ur-mom-classic">
The canonical "ur mom" joke: "ur mom so [TRAIT] that [CONSEQUENCE]".
Pick a trait from the message and exaggerate it to absurd extremes.
The consequence must be a specific, visual scenario that would
logically follow if the trait were literally true — it should
feel like creative evidence, not a random insult.
Take the trait in a surprising direction: a fat joke that becomes
a time joke, a stupid joke that becomes wordplay.
Use concrete nouns, brand names, and vivid imagery
in the consequence.
</format>
<format name="ur-mom-twist">
Take the original message and replace exactly one noun phrase
with "ur mom" with minimal other changes.
If the predicate has a sexually ambiguous word (hard, loose,
wide open, easy, dirty, down), replace the subject.
If the verb suggests physical action (fit, get into, push,
slide, squeeze), replace the object.
When the message is a question, consider answering it with
a declarative statement where "ur mom" is the answer,
echoing the original wording.
Treat the original message as raw material, not a rigid
template — flipping negation, reversing meaning, or
otherwise twisting the message is fair game if it makes
the joke land.
The fewer words changed, the better — the joke's power is
the audience recognizing the original sentence and seeing
the double meaning they missed.
</format>
<format name="twss">
"That's what she said" — reframe an innocent statement
as sexual innuendo by appending the phrase.
Only use when the message contains words with latent double
meanings: size (big, huge, long), hardness (hard, stiff, firm),
insertion (put it in, fit, squeeze), duration (keep going,
won't stop), effort (can't handle, struggling).
The wider the gap between innocent intent and sexual reading,
the funnier. The double meaning must be instantly apparent —
if it requires explanation, pick a different format.
Can be just the phrase, or a short setup echoing the message
followed by "...that's what she said."
</format>
<format name="freestyle">
Escape hatch: if the message has no trait worth exaggerating,
no natural "ur mom" substitution, and no double meaning for
TWSS — don't force it. Fall back to any joke style that fits
the context: a roast, a pun, a one-liner, an absurdist
non-sequitur, etc.
</format>
</formats>

Pick ONE format, commit to it, and deliver the joke.{russian_note}
Use wordplay, double meanings, and vivid imagery for maximum effect.
Focus on the message itself, but weave in references from
conversation context or user memories when it adds to the humor.
Make it as lewd and preposterous as possible.
Make sure that the joke is grammatically correct.

Reply in {language_name}. Return only the joke, no meta commentary or explanation.

{examples_xml}

{memories_block}

{conversation_block}"""

        async with self.telemetry.async_create_span("generate_joke"):
            response = await self._joke_writer_client.generate_content(
                message=content,
                prompt=prompt
            )
            return response

    async def generate_country_joke(self, message: str, country: str) -> str:
        prompt = f"""You are a chat bot and you need to turn a user message into a country joke.
Your response should only contain the joke itself and it should start with 'In {country}'.
Response should be fully in the language of the user message
which includes translating the country name into the user's language.
Apply stereotypes and cliches about the country."""
        async with self.telemetry.async_create_span("generate_country_joke"):
            response = await self._joke_writer_client.generate_content(
                message=message, prompt=prompt
            )
            return response

    async def is_joke(self, original_message: str, response_message: str, message_id: int = None) -> bool:
        """
        Determine if a response message is a joke to the original message.
        Includes caching to avoid redundant AI calls.
        """
        # Check cache first if message_id is provided
        if message_id is not None and message_id in self._joke_cache:
            return self._joke_cache[message_id]

        async with self.telemetry.async_create_span("is_joke", kind=SpanKind.INTERNAL):
            # Format messages in XML for clarity
            message = f"""<messages>
<original>{original_message}</original>
<response>{response_message}</response>
</messages>"""

            prompt = """Determine if the response is clearly intended
as a joke or humorous remark directed at the original message.

Only answer YES if the response is obviously humorous, a clear joke,
or deliberate wordplay. Regular conversation, even if slightly
playful or witty, should be NO."""

            logger.info("Checking if message is a joke:")
            logger.info(f"Original: {original_message}")
            logger.info(f"Response: {response_message}")

            response = await self._joke_classifier_client.generate_content(
                message=message,
                prompt=prompt,
                response_schema=YesNo
            )

            result = response.answer == "YES"

            # Cache the result if message_id is provided
            if message_id is not None:
                self._joke_cache[message_id] = result

            logger.info(f"AI response: {response.answer}")
            logger.info(f"Is joke: {result}")
            return result

    async def save_joke(self, source_message_id: int, source_message_content: str, 
                                    joke_message_id: int, joke_message_content: str, 
                                    reaction_count: int) -> None:
        """
        Save a joke to the store with language detection.
        """
        async with self.telemetry.async_create_span("save_joke"):
            # Detect languages using the language detection service
            source_lang = await self.language_detector.detect_language(source_message_content)
            joke_lang = await self.language_detector.detect_language(joke_message_content)
            
            await self.store.save(
                source_message_id=source_message_id,
                joke_message_id=joke_message_id,
                source_message_content=source_message_content,
                joke_message_content=joke_message_content,
                reaction_count=reaction_count,
                source_language=source_lang,
                joke_language=joke_lang
            )
            
            logger.info(f"Saved joke: {joke_message_content} (reactions: {reaction_count})")
