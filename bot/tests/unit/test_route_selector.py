"""Unit tests for route selection: composite fallback behaviour and prompt completeness."""

import unittest
from typing import get_args
from unittest.mock import AsyncMock, Mock

from null_telemetry import NullTelemetry
from jev_client import Choice
from route_selector import CompositeRouteSelector, JevRouteSelector, LlmRouteSelector
from schemas import RouteName, RouteSelection


class TestLlmRouteSelectorPrompt(unittest.TestCase):
    def test_prompt_offers_every_route(self):
        descriptions = {route: f"{route}: description" for route in get_args(RouteName)}
        prompt = LlmRouteSelector(Mock(), descriptions, NullTelemetry()).build_prompt("")

        for route in get_args(RouteName):
            self.assertIn(f'<route route="{route}">\n{route}: description', prompt)


class TestJevRouteSelector(unittest.IsolatedAsyncioTestCase):
    def _selector(self, probabilities: dict[str, float]) -> JevRouteSelector:
        descriptions = {route: f"{route}: description" for route in get_args(RouteName)}
        top = max(probabilities, key=probabilities.get)
        answer = Choice(choice=top, probabilities=probabilities, confidence=0.5)
        jev_client = Mock()
        jev_client.ask = AsyncMock(return_value=Mock(route=answer))
        return JevRouteSelector(jev_client, descriptions, NullTelemetry(), min_probability=0.8)

    async def test_confident_answer_is_returned(self):
        selection = await self._selector({"SCHEDULE": 0.9, "GENERAL": 0.1}).select("m", "")

        self.assertEqual(selection.route, "SCHEDULE")

    async def test_low_top_probability_is_notsure(self):
        selection = await self._selector({"GENERAL": 0.61, "SCHEDULE": 0.39}).select("m", "")

        self.assertEqual(selection.route, "NOTSURE")
        self.assertEqual(selection.reason, "jev GENERAL=0.61, SCHEDULE=0.39")


class TestCompositeRouteSelector(unittest.IsolatedAsyncioTestCase):
    def _selector(self, result: RouteSelection | Exception) -> Mock:
        selector = Mock()
        selector.select = (
            AsyncMock(side_effect=result) if isinstance(result, Exception) else AsyncMock(return_value=result)
        )
        return selector

    async def test_first_definitive_answer_wins(self):
        first = self._selector(RouteSelection(route="GENERAL", reason="jev"))
        second = self._selector(RouteSelection(route="NONE", reason="llm"))

        selection = await CompositeRouteSelector([first, second], NullTelemetry()).select("m", "")

        self.assertEqual(selection.route, "GENERAL")
        second.select.assert_not_called()

    async def test_notsure_falls_through(self):
        first = self._selector(RouteSelection(route="NOTSURE", reason="jev"))
        second = self._selector(RouteSelection(route="NONE", reason="llm"))

        selection = await CompositeRouteSelector([first, second], NullTelemetry()).select("m", "")

        self.assertEqual(selection.route, "NONE")

    async def test_exception_falls_through(self):
        first = self._selector(RuntimeError("jev down"))
        second = self._selector(RouteSelection(route="FACT", reason="llm"))

        selection = await CompositeRouteSelector([first, second], NullTelemetry()).select("m", "")

        self.assertEqual(selection.route, "FACT")

    async def test_all_unsure_or_failing_raises(self):
        first = self._selector(RuntimeError("jev down"))
        second = self._selector(RouteSelection(route="NOTSURE", reason="llm"))

        with self.assertRaises(RuntimeError):
            await CompositeRouteSelector([first, second], NullTelemetry()).select("m", "")


if __name__ == "__main__":
    unittest.main()
