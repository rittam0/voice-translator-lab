import base64
import sys
import tempfile
from pathlib import Path

from fastmcp import FastMCP

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from voice_translator.pipeline import build_pipeline

mcp = FastMCP("voice-translator")
_pipeline = None


def pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = build_pipeline()
    return _pipeline


@mcp.tool()
def translate_voice(
    audio_base64: str, source_language: str, target_language: str
) -> dict:
    """Translate a complete audio clip and return cloned speech as base64 WAV."""
    try:
        audio_bytes = base64.b64decode(audio_base64, validate=True)
    except ValueError as exc:
        raise ValueError("audio_base64 is not valid base64") from exc
    if not audio_bytes:
        raise ValueError("audio_base64 decoded to an empty clip")
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as handle:
        input_path = Path(handle.name)
        handle.write(audio_bytes)
    try:
        result = pipeline().run(input_path, source_language, target_language)
        payload = result.as_dict()
        output_path = Path(payload.pop("audio_path"))
        payload["audio_base64"] = base64.b64encode(
            output_path.read_bytes()
        ).decode("ascii")
        return payload
    finally:
        input_path.unlink(missing_ok=True)


if __name__ == "__main__":
    mcp.run()
