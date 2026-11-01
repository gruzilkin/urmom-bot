import tomllib
import unittest
from unittest.mock import AsyncMock, Mock, patch

from codex_client import _MAX_CLI_ARG_BYTES, CodexClient, _developer_instructions_arg
from null_telemetry import NullTelemetry


class TestDeveloperInstructionsArg(unittest.TestCase):
    def test_value_round_trips_through_toml(self) -> None:
        prompt = 'Line "one"\nLine two with \\backslash, tab\t, emoji 🐻 and Кириллица'
        arg = _developer_instructions_arg(prompt)
        self.assertIsNotNone(arg)
        self.assertEqual(tomllib.loads(arg)["developer_instructions"], prompt)

    def test_oversized_prompt_returns_none(self) -> None:
        self.assertIsNone(_developer_instructions_arg("x" * _MAX_CLI_ARG_BYTES))


class TestCodexInput(unittest.IsolatedAsyncioTestCase):
    async def _run(self, prompt: str) -> tuple[list[str], str]:
        """Run generate_content against a fake CLI; return the command args and the stdin text."""
        process = Mock(returncode=0)
        process.communicate = AsyncMock(return_value=(b"ok", b""))
        with patch("codex_client.asyncio.create_subprocess_exec", AsyncMock(return_value=process)) as exec_mock:
            client = CodexClient(NullTelemetry(), enable_web_search=False)
            await client.generate_content(message="<request>hi</request>", prompt=prompt)
        return list(exec_mock.await_args.args), process.communicate.await_args.kwargs["input"].decode("utf-8")

    async def test_prompt_passed_as_developer_instructions(self) -> None:
        args, stdin = await self._run("Be brief.")
        self.assertIn('developer_instructions="Be brief."', args)
        self.assertEqual(stdin, "<request>hi</request>")

    async def test_oversized_prompt_inlined_as_xml(self) -> None:
        prompt = "x" * _MAX_CLI_ARG_BYTES
        args, stdin = await self._run(prompt)
        self.assertFalse(any(arg.startswith("developer_instructions=") for arg in args))
        self.assertEqual(stdin, f"<system>{prompt}</system>\n<user_message><request>hi</request></user_message>")


if __name__ == "__main__":
    unittest.main()
