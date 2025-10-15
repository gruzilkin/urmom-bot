import unittest
from unittest.mock import AsyncMock, Mock, patch

from ai_client import AIClient
from ai_client_wrappers import CompositeAIClient, RetryAIClient
from null_telemetry import NullTelemetry


class TestRetryAIClient(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.delegate = Mock(spec=AIClient)
        self.telemetry = NullTelemetry()

    @patch("asyncio.sleep", new_callable=AsyncMock)
    async def test_passes_through_success(self, mock_sleep: AsyncMock) -> None:
        mock_sleep.return_value = None
        self.delegate.generate_content = AsyncMock(return_value="ok")
        client = RetryAIClient(self.delegate, telemetry=self.telemetry)

        result = await client.generate_content(message="hello", prompt="test")

        self.assertEqual(result, "ok")
        self.delegate.generate_content.assert_awaited_once()

    @patch("asyncio.sleep", new_callable=AsyncMock)
    async def test_retries_then_succeeds(self, mock_sleep: AsyncMock) -> None:
        mock_sleep.return_value = None
        attempt_tracker = Mock(side_effect=[RuntimeError("temporary"), "recovered"])

        async def side_effect(*args, **kwargs):
            result = attempt_tracker()
            if isinstance(result, Exception):
                raise result
            return result

        self.delegate.generate_content = AsyncMock(side_effect=side_effect)
        client = RetryAIClient(self.delegate, telemetry=self.telemetry)

        result = await client.generate_content(message="retry")

        self.assertEqual(result, "recovered")
        self.assertEqual(self.delegate.generate_content.await_count, 2)

    @patch("asyncio.sleep", new_callable=AsyncMock)
    async def test_raises_after_max_attempts(self, mock_sleep: AsyncMock) -> None:
        mock_sleep.return_value = None
        self.delegate.generate_content = AsyncMock(side_effect=ConnectionError("boom"))
        client = RetryAIClient(self.delegate, telemetry=self.telemetry)

        with self.assertRaises(ConnectionError):
            await client.generate_content(message="fail")

        self.assertEqual(self.delegate.generate_content.await_count, 3)

    @patch("asyncio.sleep", new_callable=AsyncMock)
    async def test_max_tries_explicit(self, mock_sleep: AsyncMock) -> None:
        mock_sleep.return_value = None
        self.delegate.generate_content = AsyncMock(side_effect=ConnectionError("boom"))
        client = RetryAIClient(self.delegate, telemetry=self.telemetry, max_tries=2)

        with self.assertRaises(ConnectionError):
            await client.generate_content(message="fail")

        self.assertEqual(self.delegate.generate_content.await_count, 2)

    @patch("asyncio.sleep", new_callable=AsyncMock)
    async def test_max_time_parameter(self, mock_sleep: AsyncMock) -> None:
        mock_sleep.return_value = None
        self.delegate.generate_content = AsyncMock(return_value="ok")
        client = RetryAIClient(
            self.delegate, telemetry=self.telemetry, max_time=60, jitter=True
        )

        result = await client.generate_content(message="hello")

        self.assertEqual(result, "ok")
        self.delegate.generate_content.assert_awaited_once()

    def test_both_max_time_and_max_tries_raises(self) -> None:
        with self.assertRaises(ValueError) as cm:
            RetryAIClient(
                self.delegate, telemetry=self.telemetry, max_time=60, max_tries=3
            )

        self.assertIn("Cannot specify both", str(cm.exception))


class TestCompositeAIClient(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.client_a = Mock(spec=AIClient)
        self.client_b = Mock(spec=AIClient)
        self.client_a.generate_content = AsyncMock()
        self.client_b.generate_content = AsyncMock()
        self.telemetry = NullTelemetry()

    async def test_primary_success(self) -> None:
        self.client_a.generate_content.return_value = "alpha"
        composite = CompositeAIClient(
            [self.client_a, self.client_b], telemetry=self.telemetry
        )

        result = await composite.generate_content(message="hi")

        self.assertEqual(result, "alpha")
        self.client_a.generate_content.assert_awaited_once()
        self.client_b.generate_content.assert_not_awaited()

    async def test_fallback_success(self) -> None:
        self.client_a.generate_content.side_effect = RuntimeError("boom")
        self.client_b.generate_content.return_value = "beta"
        composite = CompositeAIClient(
            [self.client_a, self.client_b], telemetry=self.telemetry
        )

        result = await composite.generate_content(message="hi")

        self.assertEqual(result, "beta")
        self.assertEqual(self.client_a.generate_content.await_count, 1)
        self.assertEqual(self.client_b.generate_content.await_count, 1)

    async def test_all_fail(self) -> None:
        self.client_a.generate_content.side_effect = ValueError("one")
        self.client_b.generate_content.side_effect = RuntimeError("two")
        composite = CompositeAIClient(
            [self.client_a, self.client_b], telemetry=self.telemetry
        )

        with self.assertRaisesRegex(RuntimeError, r"All fallback clients failed"):
            await composite.generate_content(message="hi")

        self.assertEqual(self.client_a.generate_content.await_count, 1)
        self.assertEqual(self.client_b.generate_content.await_count, 1)

    async def test_argument_passthrough(self) -> None:
        payload = {
            "message": "msg",
            "prompt": "system",
            "samples": [("u", "a")],
            "enable_grounding": True,
            "response_schema": Mock(),
            "temperature": 0.5,
            "image_data": b"data",
            "image_mime_type": "image/png",
        }

        self.client_a.generate_content.side_effect = RuntimeError("boom")
        self.client_b.generate_content.return_value = "gamma"
        composite = CompositeAIClient(
            [self.client_a, self.client_b], telemetry=self.telemetry
        )

        result = await composite.generate_content(**payload)

        self.assertEqual(result, "gamma")
        self.client_b.generate_content.assert_awaited_once_with(**payload)


if __name__ == "__main__":
    unittest.main()
