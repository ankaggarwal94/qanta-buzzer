"""Publication and ownership of the packaged T5 trainers' checkpoint directories."""

from __future__ import annotations

import re
from collections.abc import Callable
from contextlib import ExitStack
from functools import wraps
from pathlib import Path
from tempfile import TemporaryDirectory

from filelock import FileLock


def training_lock(root: str | Path) -> FileLock:
    """Serialize writers through model selection, reload, evaluation and CLI handoff."""
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return FileLock(root / ".t5-training.lock", timeout=0, is_singleton=True)


def exclusive_training_run(run):
    """Hold the output lock for a public training wrapper, including test evaluation."""

    @wraps(run)
    def wrapped(config, *args, **kwargs):
        with training_lock(config.get("checkpoint_dir", "checkpoints")):
            return run(config, *args, **kwargs)

    return wrapped


# Optional files recognized by T5EncoderModel and T5TokenizerFast can change
# how freshly saved weights or tokens are interpreted on the next load.
_T5_FILES = {
    "adapter_config.json",
    "adapter_model.bin",
    "adapter_model.safetensors",
    "added_tokens.json",
    "additional_chat_templates",
    "chat_template.jinja",
    "chat_template.json",
    "chat_templates",
    "config.json",
    "config_used.json",
    "flax_model.msgpack",
    "generation_config.json",
    "special_tokens_map.json",
    "spiece.model",
    "split_manifest.json",
    "tf_model.h5",
    "tokenizer_config.json",
}
_VERSIONED_TOKENIZER = re.compile(r"tokenizer(?:\.\d+)*\.json")
_WEIGHT_FILE = re.compile(
    r"(?:pytorch_model(?:-\d+-of-\d+)?\.bin|model(?:-\d+-of-\d+)?\.safetensors)(?:\.index\.json)?"
)


def save_training_checkpoint(
    model, destination: Path, state: dict, write_sidecars: Callable[[Path], None] | None
) -> None:
    """Publish model, optimizer and optional run metadata together after serialization."""
    import torch

    # A checkpoint directory may itself be a link to a separate volume.
    destination = destination.resolve()
    with TemporaryDirectory(dir=destination.parent, prefix=".t5-checkpoint-") as temp:
        staging = Path(temp) / "new"
        staging.mkdir()
        model.save(str(staging))
        torch.save(state, staging / "training_state.pt")
        if write_sidecars is not None:
            write_sidecars(staging)
        else:
            # Legacy comparison searches the phase directory after the model
            # directory. An explicit JSON null shadows metadata from an older CLI run.
            for filename in ("split_manifest.json", "config_used.json"):
                if (destination.parent / filename).exists():
                    (staging / filename).write_text("null\n", encoding="utf-8")

        if not destination.exists():
            staging.rename(destination)
            return

        # Preserve unrelated files (for example user notes) when replacing a
        # checkpoint. The previous model remains available if any rename fails.
        previous = Path(temp) / "previous"
        with ExitStack() as rollback:
            destination.rename(previous)
            rollback.callback(previous.rename, destination)
            for old in previous.iterdir():
                target = staging / old.name
                if (
                    target.exists()
                    or old.name in _T5_FILES
                    or _WEIGHT_FILE.fullmatch(old.name)
                    or _VERSIONED_TOKENIZER.fullmatch(old.name)
                ):
                    continue
                old.rename(target)
                rollback.callback(target.rename, old)
            staging.rename(destination)
            rollback.pop_all()
