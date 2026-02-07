import asyncio
import json
import logging
import os
import re
import shutil
import tempfile
from dataclasses import dataclass

from open_telemetry import Telemetry

logger = logging.getLogger(__name__)

SUBPROCESS_TIMEOUT_SECONDS = 300
MIN_BPP = 0.01
_CROP_PATTERN = re.compile(r"crop=(\d+):(\d+):(\d+):(\d+)")


@dataclass
class VideoInfo:
    duration: float
    width: int
    height: int
    fps: float


@dataclass
class CropBox:
    w: int
    h: int
    x: int
    y: int
    pixel_reduction: float


class VideoCompressionError(Exception):
    pass


class VideoCompressor:
    def __init__(
        self,
        telemetry: Telemetry,
        target_size_bytes: int,
        audio_bitrate_kbps: int = 64,
        ffmpeg_preset: str = "veryfast",
    ):
        self.telemetry = telemetry
        self.target_size_bytes = target_size_bytes
        self.audio_bitrate_kbps = audio_bitrate_kbps
        self.ffmpeg_preset = ffmpeg_preset

    async def _probe(self, input_path: str) -> VideoInfo:
        async with self.telemetry.async_create_span("video_compressor.probe") as span:
            process = await asyncio.create_subprocess_exec(
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                "-select_streams",
                "v:0",
                input_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=SUBPROCESS_TIMEOUT_SECONDS)
            if process.returncode != 0:
                raise VideoCompressionError(f"ffprobe failed (rc={process.returncode}): {stderr.decode()}")

            try:
                data = json.loads(stdout)
                duration = float(data["format"]["duration"])
                stream = data["streams"][0]
                width = int(stream["width"])
                height = int(stream["height"])
                r_num, r_den = stream["r_frame_rate"].split("/")
                fps = int(r_num) / int(r_den)
            except (json.JSONDecodeError, KeyError, ValueError, IndexError, ZeroDivisionError) as e:
                raise VideoCompressionError(f"ffprobe returned unparseable output: {e}") from e

            if duration <= 0:
                raise VideoCompressionError(f"ffprobe returned invalid duration: {duration}")

            span.set_attribute("duration", duration)
            span.set_attribute("width", width)
            span.set_attribute("height", height)
            span.set_attribute("fps", fps)
            return VideoInfo(duration=duration, width=width, height=height, fps=fps)

    async def _detect_crop(self, input_path: str, info: VideoInfo) -> CropBox | None:
        async with self.telemetry.async_create_span("video_compressor.detect_crop") as span:
            try:
                process = await asyncio.create_subprocess_exec(
                    "ffmpeg",
                    "-i",
                    input_path,
                    "-vf",
                    "cropdetect=limit=16:round=2:reset=0",
                    "-an",
                    "-f",
                    "null",
                    "/dev/null",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                _, stderr = await asyncio.wait_for(process.communicate(), timeout=SUBPROCESS_TIMEOUT_SECONDS)

                if process.returncode != 0:
                    logger.warning(f"cropdetect failed (rc={process.returncode})")
                    span.set_attribute("outcome", "crop_skipped")
                    return None

                matches = _CROP_PATTERN.findall(stderr.decode())
                if not matches:
                    span.set_attribute("outcome", "crop_skipped")
                    return None

                crop_tuples = [(int(w), int(h), int(x), int(y)) for w, h, x, y in matches]
                x = min(x for _, _, x, _ in crop_tuples)
                y = min(y for _, _, _, y in crop_tuples)
                w = max(x + w for w, _, x, _ in crop_tuples) - x
                h = max(y + h for _, h, _, y in crop_tuples) - y
                w -= w % 2
                h -= h % 2

                is_empty = w <= 0 or h <= 0
                is_out_of_bounds = x + w > info.width or y + h > info.height
                is_full_frame = w == info.width and h == info.height

                if is_empty or is_out_of_bounds or is_full_frame:
                    span.set_attribute("outcome", "crop_skipped")
                    return None

                pixel_reduction = 1 - (w * h) / (info.width * info.height)
                span.set_attribute("crop_w", w)
                span.set_attribute("crop_h", h)
                span.set_attribute("crop_x", x)
                span.set_attribute("crop_y", y)
                span.set_attribute("crop_pixel_reduction", round(pixel_reduction, 4))
                span.set_attribute("outcome", "crop_suggested")
                return CropBox(w=w, h=h, x=x, y=y, pixel_reduction=round(pixel_reduction, 4))

            except Exception as e:
                logger.warning(f"Crop detection failed: {e}", exc_info=True)
                span.set_attribute("outcome", "crop_skipped")
                return None

    async def analyze_crop(self, video_data: bytes) -> CropBox | None:
        tmp_fd, tmp_path = tempfile.mkstemp(prefix="vidcrop_")
        try:
            with os.fdopen(tmp_fd, "wb") as f:
                f.write(video_data)
            info = await self._probe(tmp_path)
            return await self._detect_crop(tmp_path, info)
        except Exception:
            logger.warning("Crop analysis failed", exc_info=True)
            return None
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    async def compress(self, video_data: bytes, filename: str, crop: CropBox | None = None) -> bytes | None:
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

                info = await self._probe(input_path)

                video_kbps = max(
                    1,
                    int((self.target_size_bytes * 8 * 0.9 / info.duration) / 1000 - self.audio_bitrate_kbps),
                )
                span.set_attribute("video_bitrate_kbps", video_kbps)

                uncropped_bpp = video_kbps * 1000 / (info.width * info.height * info.fps)
                if uncropped_bpp < MIN_BPP:
                    span.set_attribute("bpp", round(uncropped_bpp, 4))
                    span.set_attribute("outcome", "quality_too_low")
                    return None

                if crop is None:
                    crop = await self._detect_crop(input_path, info)

                effective_width = crop.w if crop else info.width
                effective_height = crop.h if crop else info.height

                bpp = video_kbps * 1000 / (effective_width * effective_height * info.fps)
                span.set_attribute("bpp", round(bpp, 4))

                await self._run_encode_pass(1, input_path, "/dev/null", video_kbps, passlog_prefix, crop)
                await self._run_encode_pass(2, input_path, output_path, video_kbps, passlog_prefix, crop)

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

    async def _run_encode_pass(
        self,
        pass_number: int,
        input_path: str,
        output_path: str,
        video_kbps: int,
        passlog_prefix: str,
        crop: CropBox | None,
    ) -> None:
        async with self.telemetry.async_create_span(f"video_compressor.pass{pass_number}"):
            args = ["ffmpeg", "-y", "-i", input_path]
            if crop:
                args += ["-vf", f"crop={crop.w}:{crop.h}:{crop.x}:{crop.y}"]
            args += [
                "-c:v",
                "libx264",
                "-preset",
                self.ffmpeg_preset,
                "-b:v",
                f"{video_kbps}k",
                "-pass",
                str(pass_number),
                "-passlogfile",
                passlog_prefix,
            ]
            if pass_number == 1:
                args += ["-an", "-f", "null"]
            else:
                args += ["-c:a", "aac", "-b:a", f"{self.audio_bitrate_kbps}k", "-movflags", "+faststart"]
            args += [output_path]

            process = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(process.communicate(), timeout=SUBPROCESS_TIMEOUT_SECONDS)
            if process.returncode != 0:
                raise VideoCompressionError(
                    f"ffmpeg pass {pass_number} failed (rc={process.returncode}): {stderr.decode()}"
                )
