"""Configuration loading and management utilities.

Provides functions to load YAML configurations and merge CLI overrides
using dot notation (e.g., "data.K=5" updates config["data"]["K"]).
"""

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union


def load_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
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
        # Get the project root (parent of qb_data)
        project_root = Path(__file__).parent.parent
        config_path = project_root / "configs" / "default.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


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

    return config


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
        # Load smoke config as base instead of default
        project_root = Path(__file__).parent.parent
        smoke_path = project_root / "configs" / "smoke.yaml"
        if smoke_path.exists():
            # Return special marker to load smoke config
            overrides["__config_path__"] = str(smoke_path)

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

    # Load base config
    config = load_config(config_path)

    # Apply remaining overrides
    if overrides:
        config = merge_overrides(config, overrides)

    return config


# Convenience exports
__all__ = [
    "load_config",
    "merge_overrides",
    "build_argparse_overrides",
    "add_config_args",
    "load_config_with_overrides",
]