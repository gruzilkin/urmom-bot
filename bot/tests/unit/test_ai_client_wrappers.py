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

    async def test_fallback_on_bad_response_predicate(self) -> None:
        """Composite should try the next client when the first response fails validation."""
        composite = CompositeAIClient(
            [self.client_a, self.client_b],
            telemetry=self.telemetry,
            is_bad_response=lambda resp: len(resp) > 10,
        )

        self.client_a.generate_content.return_value = "0123456789A"
        self.client_b.generate_content.return_value = "fallback"

        result = await composite.generate_content(message="check-length")

        self.assertEqual(result, "fallback")
        self.client_a.generate_content.assert_awaited_once()
        self.client_b.generate_content.assert_awaited_once()

    async def test_shuffle_false_maintains_order(self) -> None:
        """With shuffle=False, clients should always be tried in original order."""
        self.client_a.generate_content.side_effect = RuntimeError("boom")
        self.client_b.generate_content.return_value = "beta"

        composite = CompositeAIClient(
            [self.client_a, self.client_b], telemetry=self.telemetry, shuffle=False
        )

        # Run multiple times to ensure order is consistent
        for _ in range(5):
            result = await composite.generate_content(message="hi")
            self.assertEqual(result, "beta")
            # Client A should always be tried first
            self.assertEqual(self.client_a.generate_content.await_count, _ + 1)
            self.assertEqual(self.client_b.generate_content.await_count, _ + 1)

    @patch("random.shuffle")
    async def test_shuffle_true_randomizes_order(self, mock_shuffle: Mock) -> None:
        """With shuffle=True, clients should be randomized before trying."""
        self.client_a.generate_content.return_value = "alpha"

        composite = CompositeAIClient(
            [self.client_a, self.client_b], telemetry=self.telemetry, shuffle=True
        )

        result = await composite.generate_content(message="hi")

        # Shuffle should have been called once
        mock_shuffle.assert_called_once()
        # Result should still be returned correctly
        self.assertEqual(result, "alpha")

    async def test_shuffle_doesnt_break_fallback(self) -> None:
        """Shuffling should not break fallback logic."""
        client_c = Mock(spec=AIClient)
        client_c.generate_content = AsyncMock(side_effect=RuntimeError("boom"))

        self.client_a.generate_content.side_effect = RuntimeError("boom")
        self.client_b.generate_content.return_value = "success"

        composite = CompositeAIClient(
            [self.client_a, client_c, self.client_b],
            telemetry=self.telemetry,
            shuffle=True,
        )

        # Even with shuffling, one of the clients should succeed
        result = await composite.generate_content(message="hi")
        self.assertEqual(result, "success")


if __name__ == "__main__":
    unittest.main()
