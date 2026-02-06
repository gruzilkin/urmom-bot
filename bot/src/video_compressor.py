import asyncio
import json
import logging
import os
import shutil
import tempfile

from open_telemetry import Telemetry

logger = logging.getLogger(__name__)

SUBPROCESS_TIMEOUT_SECONDS = 300


class VideoCompressionError(Exception):
    pass


class VideoCompressor:
    def __init__(
        self,
        telemetry: Telemetry,
        target_size_bytes: int,
        audio_bitrate_kbps: int = 64,
        ffmpeg_preset: str = "medium",
    ):
        self.telemetry = telemetry
        self.target_size_bytes = target_size_bytes
        self.audio_bitrate_kbps = audio_bitrate_kbps
        self.ffmpeg_preset = ffmpeg_preset

    async def compress(self, video_data: bytes, filename: str) -> bytes | None:
        async with self.telemetry.async_create_span("video_compressor.compress") as span:
            span.set_attribute("input_size", len(video_data))
            span.set_attribute("filename", filename)

            temp_dir = tempfile.mkdtemp(prefix="vidcomp_")
            try:
                ext = os.path.splitext(filename)[1] or ".mp4"
                input_path = os.path.join(temp_dir, f"input{ext}")
                output_path = os.path.join(temp_dir, "output.mp4")
                passlog_prefix = os.path.join(temp_dir, "passlog")

                with open(input_path, "wb") as f:
                    f.write(video_data)

                duration = await self._probe_duration(input_path)

                video_kbps = max(
                    1,
                    int((self.target_size_bytes * 8 * 0.95 / duration) / 1000 - self.audio_bitrate_kbps),
                )
                span.set_attribute("video_bitrate_kbps", video_kbps)

                await self._run_pass1(input_path, video_kbps, passlog_prefix)
                await self._run_pass2(input_path, output_path, video_kbps, passlog_prefix)

                with open(output_path, "rb") as f:
                    output_data = f.read()

                output_size = len(output_data)
                span.set_attribute("output_size", output_size)
                span.set_attribute(
                    "compression_ratio",
                    round(len(video_data) / output_size, 2) if output_size > 0 else 0,
                )

                if output_size > self.target_size_bytes:
                    span.set_attribute("outcome", "still_too_large")
                    return None

                span.set_attribute("outcome", "success")
                return output_data

            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

    async def _probe_duration(self, input_path: str) -> float:
        async with self.telemetry.async_create_span("video_compressor.probe") as span:
            process = await asyncio.create_subprocess_exec(
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                input_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=SUBPROCESS_TIMEOUT_SECONDS
            )
            if process.returncode != 0:
                raise VideoCompressionError(f"ffprobe failed (rc={process.returncode}): {stderr.decode()}")

            try:
                info = json.loads(stdout)
                duration = float(info["format"]["duration"])
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                raise VideoCompressionError(f"ffprobe returned unparseable output: {e}") from e

            if duration <= 0:
                raise VideoCompressionError(f"ffprobe returned invalid duration: {duration}")

            span.set_attribute("duration", duration)
            return duration

    async def _run_pass1(self, input_path: str, video_kbps: int, passlog_prefix: str) -> None:
        async with self.telemetry.async_create_span("video_compressor.pass1"):
            process = await asyncio.create_subprocess_exec(
                "ffmpeg", "-y",
                "-i", input_path,
                "-c:v", "libx264",
                "-preset", self.ffmpeg_preset,
                "-b:v", f"{video_kbps}k",
                "-pass", "1",
                "-passlogfile", passlog_prefix,
                "-an",
                "-f", "null",
                "/dev/null",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(
                process.communicate(), timeout=SUBPROCESS_TIMEOUT_SECONDS
            )
            if process.returncode != 0:
                raise VideoCompressionError(f"ffmpeg pass 1 failed (rc={process.returncode}): {stderr.decode()}")

    async def _run_pass2(
        self,
        input_path: str,
        output_path: str,
        video_kbps: int,
        passlog_prefix: str,
    ) -> None:
        async with self.telemetry.async_create_span("video_compressor.pass2"):
            process = await asyncio.create_subprocess_exec(
                "ffmpeg", "-y",
                "-i", input_path,
                "-c:v", "libx264",
                "-preset", self.ffmpeg_preset,
                "-b:v", f"{video_kbps}k",
                "-pass", "2",
                "-passlogfile", passlog_prefix,
                "-c:a", "aac",
                "-b:a", f"{self.audio_bitrate_kbps}k",
                "-movflags", "+faststart",
                output_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(
                process.communicate(), timeout=SUBPROCESS_TIMEOUT_SECONDS
            )
            if process.returncode != 0:
                raise VideoCompressionError(f"ffmpeg pass 2 failed (rc={process.returncode}): {stderr.decode()}")
