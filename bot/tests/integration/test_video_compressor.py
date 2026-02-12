import os
import shutil
import tempfile
import unittest
from pathlib import Path

from null_telemetry import NullTelemetry
from video_compressor import VideoCompressor, VideoInfo

FIXTURES_DIR = Path(__file__).parent / "fixtures"
LETTERBOXED_PATH = FIXTURES_DIR / "letterboxed.mp4"
CLEAN_PATH = FIXTURES_DIR / "clean.mp4"

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


@unittest.skipUnless(FFMPEG and FFPROBE, "ffmpeg/ffprobe not found on system")
@unittest.skipUnless(LETTERBOXED_PATH.exists(), "letterboxed.mp4 not found")
@unittest.skipUnless(CLEAN_PATH.exists(), "clean.mp4 not found")
class TestVideoCompressorIntegration(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.letterboxed_data = LETTERBOXED_PATH.read_bytes()
        self.clean_data = CLEAN_PATH.read_bytes()
        self.telemetry = NullTelemetry()

    async def _probe_bytes(self, compressor: VideoCompressor, data: bytes) -> VideoInfo:
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
        try:
            with os.fdopen(tmp_fd, "wb") as f:
                f.write(data)
            return await compressor._probe(tmp_path)
        finally:
            os.unlink(tmp_path)

    async def test_probe_fixture(self):
        compressor = VideoCompressor(telemetry=self.telemetry, target_size_bytes=10 * 1024 * 1024)
        info = await self._probe_bytes(compressor, self.letterboxed_data)

        self.assertEqual(info.width, 960)
        self.assertEqual(info.height, 960)
        self.assertAlmostEqual(info.fps, 30, delta=1)
        self.assertAlmostEqual(info.duration, 37.5, delta=0.5)

    async def test_analyze_crop_detects_letterbox(self):
        compressor = VideoCompressor(telemetry=self.telemetry, target_size_bytes=10 * 1024 * 1024)
        crop = await compressor.analyze_crop(self.letterboxed_data)

        self.assertIsNotNone(crop)
        self.assertEqual(crop.w, 960)
        self.assertEqual(crop.h, 704)
        self.assertEqual(crop.x, 0)
        self.assertEqual(crop.y, 128)
        self.assertGreater(crop.pixel_reduction, 0.20)

    async def test_compress_crop_and_scale(self):
        target = 3 * 1024 * 1024
        compressor = VideoCompressor(telemetry=self.telemetry, target_size_bytes=target)
        result = await compressor.compress(self.letterboxed_data, "fixture.mp4")

        self.assertIsNotNone(result, "Expected successful compression with crop+scale")
        self.assertLessEqual(len(result), target)

        info = await self._probe_bytes(compressor, result)
        self.assertLess(info.width, 960, "Expected width to be scaled down")
        self.assertLess(info.height, 704, "Expected height to be scaled down from cropped content")

    async def test_compress_crop_only(self):
        target = 10 * 1024 * 1024
        compressor = VideoCompressor(telemetry=self.telemetry, target_size_bytes=target)
        result = await compressor.compress(self.letterboxed_data, "fixture.mp4")

        self.assertIsNotNone(result, "Expected successful compression with crop only")
        self.assertLessEqual(len(result), target)

        info = await self._probe_bytes(compressor, result)
        self.assertEqual(info.width, 960, "Expected width unchanged (crop only)")
        self.assertEqual(info.height, 704, "Expected height to match cropped content (bars removed)")

    async def test_compress_quality_too_low(self):
        target = 500 * 1024
        compressor = VideoCompressor(telemetry=self.telemetry, target_size_bytes=target)
        result = await compressor.compress(self.letterboxed_data, "fixture.mp4")

        self.assertIsNone(result, "Expected None when quality would be too low")

    async def test_compress_no_crop_no_scale(self):
        target = 10 * 1024 * 1024
        compressor = VideoCompressor(telemetry=self.telemetry, target_size_bytes=target)
        result = await compressor.compress(self.clean_data, "clean.mp4")

        self.assertIsNotNone(result, "Expected successful compression with no crop or scale")
        self.assertLessEqual(len(result), target)

        info = await self._probe_bytes(compressor, result)
        self.assertEqual(info.width, 960, "Expected width unchanged")
        self.assertEqual(info.height, 704, "Expected height unchanged")

    async def test_compress_scale_only(self):
        target = 3 * 1024 * 1024
        compressor = VideoCompressor(telemetry=self.telemetry, target_size_bytes=target)
        result = await compressor.compress(self.clean_data, "clean.mp4")

        self.assertIsNotNone(result, "Expected successful compression with scale only")
        self.assertLessEqual(len(result), target)

        info = await self._probe_bytes(compressor, result)
        self.assertLess(info.width, 960, "Expected width to be scaled down")
        self.assertLess(info.height, 704, "Expected height to be scaled down")


if __name__ == "__main__":
    unittest.main()
