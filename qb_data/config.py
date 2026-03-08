"""Configuration loading and management utilities.

Provides functions to load YAML configurations, apply small
cross-codebase compatibility normalizations, and merge CLI overrides
using dot notation (e.g., ``data.K=5`` updates ``config["data"]["K"]``).
"""

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Union


def normalize_config(
    config: Dict[str, Any],
    smoke: bool = False,
) -> Dict[str, Any]:
    """Apply compatibility defaults to a loaded configuration.

    Parameters
    ----------
    config : dict
        Parsed configuration dictionary.
    smoke : bool
        Whether the caller intends to run in smoke mode.

    Returns
    -------
    dict
        Normalized configuration dictionary.
    """
    data_cfg = config.setdefault("data", {})
    env_cfg = config.setdefault("environment", {})
    lik_cfg = config.setdefault("likelihood", {})

    if "reward" in env_cfg and "reward_mode" not in env_cfg:
        env_cfg["reward_mode"] = env_cfg["reward"]
    elif "reward_mode" in env_cfg and "reward" not in env_cfg:
        env_cfg["reward"] = env_cfg["reward_mode"]

    if smoke and data_cfg.get("dataset_smoke") and "dataset" not in data_cfg:
        data_cfg["dataset"] = data_cfg["dataset_smoke"]
    if smoke and data_cfg.get("dataset_smoke_config") and "dataset_config" not in data_cfg:
        data_cfg["dataset_config"] = data_cfg["dataset_smoke_config"]

    if "embedding_model" in lik_cfg and "sbert_name" not in lik_cfg:
        lik_cfg["sbert_name"] = lik_cfg["embedding_model"]
    if "sbert_name" in lik_cfg and "embedding_model" not in lik_cfg:
        lik_cfg["embedding_model"] = lik_cfg["sbert_name"]

    return config


def resolve_data_loading_options(
    config: Dict[str, Any],
    smoke: bool = False,
) -> Dict[str, Any]:
    """Resolve CSV/Hugging Face data-loading options from a config dict.

    Parameters
    ----------
    config : dict
        Parsed configuration dictionary.
    smoke : bool
        Whether the caller intends to run in smoke mode.

    Returns
    -------
    dict
        Resolved data-loading settings.
    """
    data_cfg = config.get("data", {})
    use_smoke_dataset = smoke and any(
        data_cfg.get(key) is not None
        for key in ("dataset_smoke", "dataset_smoke_config", "split_smoke", "csv_smoke_path")
    )

    csv_path = data_cfg.get("csv_path")
    if smoke and data_cfg.get("csv_smoke_path"):
        csv_path = data_cfg["csv_smoke_path"]

    dataset = data_cfg.get("dataset")
    dataset_config = data_cfg.get("dataset_config")
    split = data_cfg.get("split", "eval")

    if use_smoke_dataset:
        dataset = data_cfg.get("dataset_smoke", dataset)
        dataset_config = data_cfg.get("dataset_smoke_config", dataset_config)
        split = data_cfg.get("split_smoke", split)

    return {
        "csv_path": csv_path,
        "dataset": dataset,
        "dataset_config": dataset_config,
        "split": split,
        "use_huggingface": bool(data_cfg.get("use_huggingface", False) or dataset),
        "max_questions": data_cfg.get("max_questions"),
        "uses_dataset_smoke": use_smoke_dataset,
    }


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    smoke: bool = False,
) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to configuration file. Defaults to configs/default.yaml.

    Returns
    -------
    dict
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If config file doesn't exist.
    ImportError
        If PyYAML is not installed.
    """
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required for config loading. "
            "Install it with: pip install pyyaml"
        )

    # Default to configs/default.yaml if no path given
    if config_path is None:
        project_root = Path(__file__).parent.parent
        default_path = project_root / "configs" / "default.yaml"
        smoke_path = project_root / "configs" / "smoke.yaml"

        if smoke and default_path.exists():
            with open(default_path, "r", encoding="utf-8") as f:
                default_config = yaml.safe_load(f) or {}
            default_data = default_config.get("data", {})
            if any(
                default_data.get(key) is not None
                for key in ("dataset_smoke", "dataset_smoke_config", "split_smoke", "csv_smoke_path")
            ):
                config_path = default_path
            elif smoke_path.exists():
                config_path = smoke_path
            else:
                config_path = default_path
        else:
            config_path = default_path
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return normalize_config(config or {}, smoke=smoke)


def merge_overrides(
    config: Dict[str, Any],
    overrides: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge override values into configuration using dot notation.

    Parameters
    ----------
    config : dict
        Base configuration dictionary.
    overrides : dict
        Override values to merge. Keys can use dot notation
        (e.g., {"data.K": 5} updates config["data"]["K"]).

    Returns
    -------
    dict
        Updated configuration with overrides applied.

    Examples
    --------
    >>> config = {"data": {"K": 4}, "ppo": {"batch_size": 32}}
    >>> overrides = {"data.K": 5, "ppo.batch_size": 16}
    >>> config = merge_overrides(config, overrides)
    >>> assert config["data"]["K"] == 5
    >>> assert config["ppo"]["batch_size"] == 16
    """
    for key, value in overrides.items():
        # Split on dots for nested keys
        keys = key.split(".")

        # Navigate to the nested location
        current = config
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        # Set the final value
        final_key = keys[-1]
        current[final_key] = value

    return normalize_config(config)


def build_argparse_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """Convert argparse namespace to configuration overrides.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    dict
        Configuration overrides extracted from args.

    Notes
    -----
    Special handling:
    - --smoke flag loads smoke.yaml config
    - --config specifies custom config path
    - --override key=value pairs for dot notation overrides
    """
    overrides = {}

    # Handle smoke test mode
    if hasattr(args, "smoke") and args.smoke:
        overrides["__smoke__"] = True

    # Handle custom config path
    if hasattr(args, "config") and args.config:
        overrides["__config_path__"] = args.config

    # Parse key=value override pairs
    if hasattr(args, "override") and args.override:
        for override_str in args.override:
            if "=" not in override_str:
                print(f"Warning: Invalid override format '{override_str}', expected 'key=value'")
                continue

            key, value_str = override_str.split("=", 1)

            # Try to parse value as appropriate type
            value = parse_value(value_str)
            overrides[key] = value

    return overrides


def parse_value(value_str: str) -> Any:
    """Parse string value to appropriate Python type.

    Parameters
    ----------
    value_str : str
        String representation of value.

    Returns
    -------
    any
        Parsed value with appropriate type.

    Examples
    --------
    >>> parse_value("5") == 5
    >>> parse_value("3.14") == 3.14
    >>> parse_value("true") == True
    >>> parse_value("false") == False
    >>> parse_value("null") == None
    >>> parse_value("hello") == "hello"
    """
    # Handle boolean values
    if value_str.lower() == "true":
        return True
    if value_str.lower() == "false":
        return False

    # Handle null/none
    if value_str.lower() in ("null", "none"):
        return None

    # Try to parse as number
    try:
        # Try integer first
        if "." not in value_str:
            return int(value_str)
        # Then float
        return float(value_str)
    except ValueError:
        pass

    # Return as string
    return value_str


def add_config_args(parser: argparse.ArgumentParser) -> None:
    """Add configuration-related arguments to parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to add arguments to.
    """
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use smoke test configuration for quick testing"
    )
    parser.add_argument(
        "--override",
        action="append",
        help="Override config values using dot notation (e.g., data.K=5)"
    )


def load_config_with_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """Load configuration and apply command-line overrides.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    dict
        Final configuration with all overrides applied.
    """
    # Build overrides from args
    overrides = build_argparse_overrides(args)

    # Check for special config path
    config_path = overrides.pop("__config_path__", None)
    smoke = bool(overrides.pop("__smoke__", False))

    # Load base config
    config = load_config(config_path, smoke=smoke)

    # Apply remaining overrides
    if overrides:
        config = merge_overrides(config, overrides)

    return config


# Convenience exports
__all__ = [
    "load_config",
    "merge_overrides",
    "normalize_config",
    "resolve_data_loading_options",
    "build_argparse_overrides",
    "add_config_args",
    "load_config_with_overrides",
]
