# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2026 Argmax, Inc. All Rights Reserved.

"""ArgmaxPrototypes `tts-cli` engine — voice-cloning-capable text-to-speech.

Unlike `ArgmaxOpenSourceEngine` (which clones and `swift build`s the Swift
`argmax-cli`), `tts-cli` is a Python console script installed from the
`argmax_prototypes` package (see pyproject `[tool.uv.sources]`). We therefore
just resolve the executable (default: `tts-cli` on PATH) and invoke it as a
subprocess.

`tts-cli` writes both `{output_dir}/{output_filename}.wav` and a `.npy` of the
RVQ frames; this engine returns the path to the WAV.
"""

import shutil
import subprocess
import sys
from pathlib import Path

from argmaxtools.utils import get_logger
from pydantic import BaseModel, Field


logger = get_logger(__name__)

DEFAULT_TTS_CLI = "tts-cli"

# ArgmaxPrototypes can't be installed into OpenBench's venv (it depends on
# openbench itself — a circular dependency — plus heavy internal/research deps),
# so `tts-cli` is run from the prototype's OWN venv as a subprocess. The console
# script's shebang points at that venv's Python, so it brings its own correct
# dependencies (e.g. the internal argmaxtools). These are the conventional
# locations of that venv; override with `cli_path` if yours lives elsewhere.
def _candidate_prototype_clis() -> list[Path]:
    home = Path.home()
    # OpenBench repo root, to also look for a sibling ArgmaxPrototypes checkout.
    repo_root = Path(__file__).resolve().parents[3]
    bases = [
        home / "Projects" / "ArgmaxPrototypes",
        repo_root.parent / "ArgmaxPrototypes",
        home / "ArgmaxPrototypes",
    ]
    return [base / ".venv" / "bin" / "tts-cli" for base in bases]


def _resolve_tts_cli(cli: str) -> str | None:
    """Resolve the `tts-cli` executable.

    Tries, in order: an explicit/expanded path; `cli` on PATH; the console
    script next to the running interpreter (co-installed venv); and finally the
    conventional ArgmaxPrototypes venv locations (the common case, since the
    prototype lives in its own venv).
    """
    expanded = Path(cli).expanduser()
    if expanded.exists():
        return str(expanded.resolve())
    on_path = shutil.which(cli)
    if on_path is not None:
        return on_path
    sibling = Path(sys.executable).parent / cli
    if sibling.exists():
        return str(sibling.resolve())
    # Only fall back to prototype-venv discovery when using the default name
    # (an explicit cli_path that doesn't exist should surface as an error).
    if cli == DEFAULT_TTS_CLI:
        for candidate in _candidate_prototype_clis():
            if candidate.exists():
                return str(candidate.resolve())
    return None


class ArgmaxPrototypeEngineConfig(BaseModel):
    """Engine config: where to find the `tts-cli` executable."""

    cli_path: str | None = Field(
        default=None,
        description=(
            "Path to the `tts-cli` executable. If unset, it is auto-discovered: PATH, then the "
            "running venv, then conventional ArgmaxPrototypes venv locations "
            "(e.g. ~/Projects/ArgmaxPrototypes/.venv/bin/tts-cli)."
        ),
    )


class PrototypeTtsInput(BaseModel):
    """Input for a single `tts-cli` synthesis call."""

    text: str = Field(..., description="Text to synthesize.")
    output_dir: Path = Field(..., description="Directory `tts-cli` writes outputs into.")
    output_filename: str = Field(..., description="Base filename (no extension) for the generated audio.")


class PrototypeTtsOutput(BaseModel):
    """Output from a `tts-cli` synthesis call."""

    audio_path: Path = Field(..., description="Path to the generated WAV file.")


class ArgmaxPrototypeEngine:
    """Resolve `tts-cli` and run text-to-speech synthesis with a prebuilt flag list."""

    def __init__(self, config: ArgmaxPrototypeEngineConfig) -> None:
        self.config = config
        cli = config.cli_path or DEFAULT_TTS_CLI
        resolved = _resolve_tts_cli(cli)
        if resolved is None:
            raise RuntimeError(
                f"Could not find the `tts-cli` executable ({cli!r}). Set up the ArgmaxPrototypes "
                "venv (e.g. at ~/Projects/ArgmaxPrototypes with `uv sync`), or set `cli_path` to its "
                "`tts-cli` (e.g. ~/Projects/ArgmaxPrototypes/.venv/bin/tts-cli)."
            )
        self.cli_path = resolved
        logger.info("Using ArgmaxPrototypes tts-cli at %s", self.cli_path)

    def tts(self, input: PrototypeTtsInput, tts_args: list[str]) -> PrototypeTtsOutput:
        """Run `tts-cli` with a pre-built flag list (see the prototype pipeline config)."""
        input.output_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            self.cli_path,
            "--text",
            input.text,
            "--output-dir",
            str(input.output_dir),
            "--output-filename",
            input.output_filename,
            *tts_args,
        ]
        logger.debug("ArgmaxPrototypes tts-cli: %s", cmd)
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"tts-cli failed: {e.stderr}") from e

        audio_path = input.output_dir / f"{input.output_filename}.wav"
        if not audio_path.is_file():
            raise RuntimeError(f"tts-cli reported success but no WAV was written at {audio_path}")
        return PrototypeTtsOutput(audio_path=audio_path)
