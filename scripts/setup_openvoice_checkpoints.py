from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

OFFICIAL_REVISION = "fd981100305a0e4291f93a9ad169c6d9f7bed54a"
OFFICIAL_BASE_URL = (
    "https://huggingface.co/myshell-ai/OpenVoiceV2/resolve/"
    f"{OFFICIAL_REVISION}/converter"
)


def validate(root: Path) -> dict[str, int]:
    converter = root / "converter"
    required = {
        "config.json": converter / "config.json",
        "checkpoint.pth": converter / "checkpoint.pth",
    }
    for name, path in required.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty OpenVoice file: {name}")
    with required["config.json"].open(encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict) or not config:
        raise RuntimeError("OpenVoice config.json is not a non-empty JSON object")
    if required["checkpoint.pth"].stat().st_size < 1_000_000:
        raise RuntimeError("OpenVoice checkpoint.pth is unexpectedly small")
    return {name: path.stat().st_size for name, path in required.items()}


def safe_extract(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if destination.resolve() not in target.parents and target != destination.resolve():
                raise RuntimeError("Unsafe path in checkpoint archive")
        bundle.extractall(destination)


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(
        url, headers={"User-Agent": "voice-translator-checkpoint-setup/1.0"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        with destination.open("wb") as output:
            shutil.copyfileobj(response, output)


def install(destination: Path, base_url: str) -> dict[str, int]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="openvoice-download-") as temporary:
        temporary_path = Path(temporary)
        source_root = temporary_path / "checkpoints_v2"
        converter = source_root / "converter"
        converter.mkdir(parents=True)
        _download(f"{base_url}/config.json?download=true", converter / "config.json")
        _download(
            f"{base_url}/checkpoint.pth?download=true",
            converter / "checkpoint.pth",
        )
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source_root, destination)
    return validate(destination)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and validate official OpenVoice V2 converter files."
    )
    parser.add_argument(
        "--destination",
        default="checkpoints_v2",
        type=Path,
        help="Ignored OpenVoice checkpoint directory",
    )
    parser.add_argument(
        "--base-url",
        default=OFFICIAL_BASE_URL,
        help="Pinned official myshell-ai/OpenVoiceV2 converter directory",
    )
    parser.add_argument(
        "--validate-only", action="store_true", help="Do not download"
    )
    args = parser.parse_args()
    sizes = (
        validate(args.destination)
        if args.validate_only
        else install(args.destination, args.base_url)
    )
    print(f"OpenVoice V2 converter ready at {args.destination.resolve()}")
    for name, size in sizes.items():
        print(f"{name}: {size} bytes")


if __name__ == "__main__":
    main()
