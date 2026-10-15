import unittest
from datetime import date, datetime, timezone
from unittest.mock import AsyncMock, patch
from zoneinfo import ZoneInfo

from conversation_formatter import format_timestamp
from store import GuildConfig, local_day_bounds
from test_store import TestStore


class TestLocalDayBounds(unittest.TestCase):
    def test_tokyo_day_starts_at_previous_utc_afternoon(self) -> None:
        start, end = local_day_bounds(date(2026, 10, 15), ZoneInfo("Asia/Tokyo"))

        self.assertEqual(start, datetime(2026, 10, 14, 15, 0, tzinfo=timezone.utc))
        self.assertEqual(end, datetime(2026, 10, 15, 15, 0, tzinfo=timezone.utc))

    def test_dst_start_day_is_23_hours(self) -> None:
        start, end = local_day_bounds(date(2026, 3, 8), ZoneInfo("America/Los_Angeles"))

        self.assertEqual(start, datetime(2026, 3, 8, 8, 0, tzinfo=timezone.utc))
        self.assertEqual(end, datetime(2026, 3, 9, 7, 0, tzinfo=timezone.utc))

    def test_utc_day_is_unchanged(self) -> None:
        start, end = local_day_bounds(date(2026, 10, 15), ZoneInfo("UTC"))

        self.assertEqual(start, datetime(2026, 10, 15, 0, 0, tzinfo=timezone.utc))
        self.assertEqual(end, datetime(2026, 10, 16, 0, 0, tzinfo=timezone.utc))


class TestFormatTimestamp(unittest.TestCase):
    def test_renders_local_time_with_weekday_and_offset(self) -> None:
        timestamp = datetime(2026, 10, 15, 12, 42, tzinfo=timezone.utc)

        self.assertEqual(format_timestamp(timestamp, ZoneInfo("Asia/Tokyo")), "2026-10-15 Thu 21:42 +09:00")

    def test_negative_offset_crosses_into_previous_day(self) -> None:
        timestamp = datetime(2026, 10, 15, 2, 5, tzinfo=timezone.utc)

        self.assertEqual(format_timestamp(timestamp, ZoneInfo("America/Los_Angeles")), "2026-10-14 Wed 19:05 -07:00")


class TestGetGuildTimezone(unittest.IsolatedAsyncioTestCase):
    async def test_returns_configured_zone(self) -> None:
        store = TestStore()
        with patch.object(
            store, "get_guild_config", AsyncMock(return_value=GuildConfig(guild_id=1, default_timezone="Asia/Tokyo"))
        ):
            self.assertEqual(await store.get_guild_timezone(1), ZoneInfo("Asia/Tokyo"))

    async def test_invalid_zone_falls_back_to_utc(self) -> None:
        store = TestStore()
        with (
            patch.object(
                store, "get_guild_config", AsyncMock(return_value=GuildConfig(guild_id=1, default_timezone="Made/Up"))
            ),
            self.assertLogs("store", level="ERROR"),
        ):
            self.assertEqual(await store.get_guild_timezone(1), ZoneInfo("UTC"))


if __name__ == "__main__":
    unittest.main()
