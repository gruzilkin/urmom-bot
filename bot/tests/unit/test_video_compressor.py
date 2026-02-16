import unittest

from video_compressor import _filter_crop_outliers, _CROP_MIN_FRAMES


class TestFilterCropOutliers(unittest.TestCase):
    """Tests for frequency-based crop outlier removal."""

    def test_stable_frames_returns_single_value(self):
        tuples = [(704, 480, 12, 418)] * 100
        result = _filter_crop_outliers(tuples)

        self.assertEqual(result, [(704, 480, 12, 418)])

    def test_frames_at_threshold_kept(self):
        base = [(704, 492, 12, 418)] * 100
        borderline = [(704, 496, 12, 415)] * _CROP_MIN_FRAMES

        result = _filter_crop_outliers(base + borderline)

        self.assertEqual(len(result), 2)
        self.assertIn((704, 492, 12, 418), result)
        self.assertIn((704, 496, 12, 415), result)

    def test_mixed_frequencies(self):
        """Only values above the threshold survive."""
        common_a = [(700, 490, 12, 420)] * 50
        common_b = [(702, 492, 10, 418)] * 30
        rare = [(718, 906, 0, 0)] * 3

        result = _filter_crop_outliers(common_a + common_b + rare)

        self.assertEqual(len(result), 2)
        self.assertIn((700, 490, 12, 420), result)
        self.assertIn((702, 492, 10, 418), result)


if __name__ == "__main__":
    unittest.main()
