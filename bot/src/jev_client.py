"""Typed client for TypeSafe AI's Jev "System One" model.

Jev does not generate text. It evaluates a state against typed questions and returns a calibrated
probability distribution per question. Questions are declared as pydantic models whose fields are
`Choice[...]`, `Noul`, or `Score` and are described with the `choice()`, `noul()`, and `score()`
field helpers. `JevClient.ask()` sends every field of the model as one request and returns an
instance of the model populated with the answers.

    class RouteDecision(BaseModel):
        route: Choice[Literal["GENERAL", "NONE"]] = choice(
            instructions="How should this message be handled?",
            criteria={"GENERAL": "A question for the bot", "NONE": "Small talk"},
        )

    decision = await jev.ask(state={"message": text}, questions=RouteDecision)
    decision.route.choice          # "GENERAL"
    decision.route.probabilities   # {"GENERAL": 0.93, "NONE": 0.07}
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from functools import cache
from typing import Any, Generic, Literal, TypeVar, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy
from typesafe_sdk import Choice as SdkChoice
from typesafe_sdk import Noul as SdkNoul
from typesafe_sdk import Score as SdkScore
from typesafe_sdk import ChoiceAnswer as SdkChoiceAnswer
from typesafe_sdk import NoulAnswer as SdkNoulAnswer
from typesafe_sdk import ScoreAnswer as SdkScoreAnswer

from open_telemetry import Telemetry

logger = logging.getLogger(__name__)

SERVICE_NAME = "JEV"

LabelT = TypeVar("LabelT", bound=str)
QuestionsT = TypeVar("QuestionsT", bound=BaseModel)

JsonContent = str | dict[str, Any] | list[Any]


class Choice(BaseModel, Generic[LabelT]):
    """Answer to a choice question: the selected label plus the full distribution."""

    model_config = ConfigDict(frozen=True)

    choice: LabelT
    probabilities: dict[str, float]
    confidence: float

    @property
    def top_probability(self) -> float:
        return self.probabilities.get(self.choice, max(self.probabilities.values()))


class Noul(BaseModel):
    """Answer to a yes/no question: the probability that the statement is true."""

    model_config = ConfigDict(frozen=True)

    probability: float


class Score(BaseModel):
    """Answer to a score question: the expected level plus the distribution over levels."""

    model_config = ConfigDict(frozen=True)

    score: float
    probabilities: dict[int, float]
    legend: dict[int, str]
    confidence: float


_SPEC_KEY = "jev"


def choice(instructions: str, criteria: Mapping[str, str | None] | Iterable[str] | None = None) -> Any:
    """Declare a `Choice` field. Labels come from `criteria` when given, else from the `Literal` type."""
    if criteria is not None and not isinstance(criteria, Mapping):
        criteria = {label: None for label in criteria}
    return Field(json_schema_extra={_SPEC_KEY: {"type": "choice", "instructions": instructions, "criteria": criteria}})


def noul(instructions: str, true: str | None = None, false: str | None = None) -> Any:
    """Declare a `Noul` field with optional descriptions of the yes and no outcomes."""
    return Field(
        json_schema_extra={_SPEC_KEY: {"type": "noul", "instructions": instructions, "true": true, "false": false}}
    )


def score(instructions: str, levels: Sequence[str]) -> Any:
    """Declare a `Score` field. Level index is the score, starting at zero."""
    return Field(json_schema_extra={_SPEC_KEY: {"type": "score", "instructions": instructions, "levels": list(levels)}})


def _generic_origin_and_args(annotation: Any) -> tuple[Any, tuple[Any, ...]]:
    metadata = getattr(annotation, "__pydantic_generic_metadata__", None)
    if metadata and metadata.get("origin") is not None:
        return metadata["origin"], tuple(metadata["args"])
    return annotation, ()


def _literal_labels(args: tuple[Any, ...]) -> tuple[str, ...] | None:
    if len(args) == 1 and get_origin(args[0]) is Literal:
        return tuple(get_args(args[0]))
    return None


def _build_question(
    model: type[BaseModel], name: str, annotation: Any, spec: dict[str, Any]
) -> SdkChoice | SdkNoul | SdkScore:
    origin, args = _generic_origin_and_args(annotation)
    kind = spec["type"]

    if kind == "choice":
        if origin is not Choice:
            raise TypeError(f"{model.__name__}.{name}: choice() requires a Choice[...] annotation")
        labels = _literal_labels(args)
        criteria = spec["criteria"]
        if criteria is None:
            if labels is None:
                raise TypeError(
                    f"{model.__name__}.{name}: choice() needs criteria or a Choice[Literal[...]] annotation"
                )
            criteria = {label: None for label in labels}
        elif labels is not None and set(criteria) != set(labels):
            raise TypeError(
                f"{model.__name__}.{name}: criteria labels {sorted(criteria)} do not match Literal {sorted(labels)}"
            )
        return SdkChoice(instructions=spec["instructions"], criteria=dict(criteria))

    if kind == "noul":
        if origin is not Noul:
            raise TypeError(f"{model.__name__}.{name}: noul() requires a Noul annotation")
        criteria = {key: spec[key] for key in ("true", "false") if spec[key] is not None}
        return SdkNoul(instructions=spec["instructions"], criteria=criteria or None)

    if kind == "score":
        if origin is not Score:
            raise TypeError(f"{model.__name__}.{name}: score() requires a Score annotation")
        return SdkScore(instructions=spec["instructions"], criteria=spec["levels"])

    raise TypeError(f"{model.__name__}.{name}: unknown question type {kind!r}")


@cache
def build_questions(model: type[BaseModel]) -> dict[str, SdkChoice | SdkNoul | SdkScore]:
    """Translate every field of a question model into SDK question objects. Cached per model class."""
    questions: dict[str, SdkChoice | SdkNoul | SdkScore] = {}
    for name, info in model.model_fields.items():
        extra = info.json_schema_extra if isinstance(info.json_schema_extra, dict) else {}
        spec = extra.get(_SPEC_KEY)
        if not isinstance(spec, dict):
            raise TypeError(f"{model.__name__}.{name}: declare Jev fields with choice(), noul(), or score()")
        questions[name] = _build_question(model, name, info.annotation, spec)
    if not questions:
        raise TypeError(f"{model.__name__} declares no questions")
    return questions


def _convert_answer(annotation: Any, answer: SdkChoiceAnswer | SdkNoulAnswer | SdkScoreAnswer) -> Choice | Noul | Score:
    if isinstance(answer, SdkChoiceAnswer):
        return annotation(choice=answer.choice, probabilities=dict(answer.probabilities), confidence=answer.confidence)
    if isinstance(answer, SdkNoulAnswer):
        return Noul(probability=answer.noul)
    if isinstance(answer, SdkScoreAnswer):
        legend = {level: text if isinstance(text, str) else str(text) for level, text in answer.legend.items()}
        return Score(
            score=answer.score, probabilities=dict(answer.probabilities), legend=legend, confidence=answer.confidence
        )
    raise TypeError(f"Unsupported Jev answer type: {type(answer).__name__}")


class JevClient:
    """Sends a pydantic question model to Jev and returns it populated with typed answers."""

    def __init__(
        self,
        api_key: str,
        telemetry: Telemetry,
        model: str = "jev-latest",
        timeout_seconds: float = 5.0,
        max_retries: int = 1,
    ) -> None:
        self.telemetry = telemetry
        self.model_name = model
        self._client = AsyncTypeSafeClient(
            api_key=api_key,
            model=model,
            timeout=timeout_seconds,
            retry=RetryPolicy(max_retries=max_retries),
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.model_name})"

    async def ask(self, state: JsonContent, questions: type[QuestionsT]) -> QuestionsT:
        """Evaluate `state` against every question declared on `questions` in a single request."""
        wire_questions = build_questions(questions)

        async with self.telemetry.async_create_span("jev_ask") as span:
            span.set_attribute("service", SERVICE_NAME)
            span.set_attribute("model", self.model_name)
            span.set_attribute("questions", ",".join(wire_questions))

            response = await self._client.system_one(state=state, questions=wire_questions)

            span.set_attribute("response_model", response.model)
            self.telemetry.track_token_usage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=None,
                attributes={"service": SERVICE_NAME, "model": response.model},
            )

            answers: dict[str, Choice | Noul | Score] = {}
            for name, info in questions.model_fields.items():
                if name not in response.answers:
                    raise ValueError(f"Jev response is missing an answer for question {name!r}")
                answers[name] = _convert_answer(info.annotation, response.answers[name])
                span.set_attribute(f"answer.{name}", _summarize(answers[name]))

            return questions(**answers)

    async def aclose(self) -> None:
        await self._client.aclose()


def _summarize(answer: Choice | Noul | Score) -> str:
    if isinstance(answer, Choice):
        return f"{answer.choice} ({answer.top_probability:.2f}, confidence {answer.confidence:.2f})"
    if isinstance(answer, Noul):
        return f"{answer.probability:.2f}"
    return f"{answer.score:.2f} (confidence {answer.confidence:.2f})"
