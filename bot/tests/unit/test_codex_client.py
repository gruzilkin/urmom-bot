"""Unit tests for CodexClient JSON-schema strictness handling."""

import unittest

from codex_client import _enforce_no_additional_properties
from schemas import DailySummaries, YesNo


class TestCodexSchemaStrictness(unittest.TestCase):
    """Codex strict structured output requires additionalProperties:false on every object."""

    def test_nested_objects_get_additional_properties_false(self) -> None:
        """A schema with nested models ($defs) must forbid extra props at every level."""
        schema = DailySummaries.model_json_schema()
        nested_objects = [d for d in schema.get("$defs", {}).values() if d.get("type") == "object"]
        # Guard the test itself: DailySummaries must actually have a nested object (UserSummary).
        self.assertTrue(nested_objects, "expected DailySummaries to contain a nested $defs object")

        _enforce_no_additional_properties(schema)

        self.assertIs(schema["additionalProperties"], False)
        for name, definition in schema["$defs"].items():
            if definition.get("type") == "object":
                self.assertIs(
                    definition["additionalProperties"],
                    False,
                    f"$defs.{name} must forbid additional properties",
                )

    def test_flat_schema_marked(self) -> None:
        """A flat schema still gets the top-level flag."""
        schema = YesNo.model_json_schema()
        _enforce_no_additional_properties(schema)
        self.assertIs(schema["additionalProperties"], False)


if __name__ == "__main__":
    unittest.main()
