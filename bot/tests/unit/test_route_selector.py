"""Unit tests for route selection: composite fallback behaviour and prompt completeness."""

import unittest
from typing import get_args
from unittest.mock import AsyncMock, Mock

from null_telemetry import NullTelemetry
from route_selector import CompositeRouteSelector, LlmRouteSelector
from schemas import RouteName, RouteSelection


class TestLlmRouteSelectorPrompt(unittest.TestCase):
    def test_prompt_offers_every_route(self):
        descriptions = {route: f"{route}: description" for route in get_args(RouteName)}
        prompt = LlmRouteSelector(Mock(), descriptions, NullTelemetry()).build_prompt("")

        for route in get_args(RouteName):
            self.assertIn(f'<route route="{route}">\n{route}: description', prompt)


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
