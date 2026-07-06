"""
utils/config_loader.py
======================
Type-safe configuration loader for the Villu Pattu Tala Identification System.
Reads config/config.yaml and provides a validated, dot-accessible config object.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (override wins)."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


class _AttrDict(dict):
    """A dict subclass that allows attribute-style access."""

    def __getattr__(self, key: str) -> Any:
        try:
            val = self[key]
            if isinstance(val, dict):
                return _AttrDict(val)
            return val
        except KeyError:
            raise AttributeError(f"Config has no key '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def get_path(self, *keys: str, root: Optional[Path] = None) -> Path:
        """Retrieve a value as a Path, optionally prefixed by *root*."""
        val = self
        for k in keys:
            val = val[k]
        p = Path(str(val))
        if root is not None and not p.is_absolute():
            return root / p
        return p


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"
_PROJECT_ROOT = Path(__file__).parent.parent


def load_config(
    config_path: Optional[str | Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> _AttrDict:
    """Load and return the project configuration.

    Parameters
    ----------
    config_path:
        Path to the YAML config file.  Defaults to ``config/config.yaml``
        relative to the project root.
    overrides:
        Optional flat dict of overrides (supports nested dot keys, e.g.
        ``{'audio.sample_rate': 44100}``).

    Returns
    -------
    _AttrDict
        Validated, attribute-accessible config object.
    """
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg: dict = yaml.safe_load(f)

    # Apply env-variable overrides (prefix VILLU_)
    cfg = _apply_env_overrides(cfg)

    # Apply programmatic overrides
    if overrides:
        cfg = _apply_dot_overrides(cfg, overrides)

    # Inject a convenience ``paths`` section with resolved absolute paths
    cfg["_project_root"] = str(_PROJECT_ROOT)
    cfg = _resolve_paths(cfg)

    return _AttrDict(cfg)


def _apply_env_overrides(cfg: dict) -> dict:
    """Apply environment variables prefixed with ``VILLU_``."""
    for env_key, env_val in os.environ.items():
        if not env_key.startswith("VILLU_"):
            continue
        # VILLU_AUDIO__SAMPLE_RATE → audio.sample_rate
        dot_key = env_key[6:].lower().replace("__", ".")
        cfg = _apply_dot_overrides(cfg, {dot_key: _coerce(env_val)})
    return cfg


def _apply_dot_overrides(cfg: dict, overrides: dict) -> dict:
    """Set ``a.b.c`` style keys into the nested config dict."""
    for dot_key, value in overrides.items():
        keys = dot_key.split(".")
        d = cfg
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
    return cfg


def _coerce(value: str) -> Any:
    """Attempt to parse a string env var into a Python scalar."""
    if value.lower() in ("true", "yes"):
        return True
    if value.lower() in ("false", "no"):
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _resolve_paths(cfg: dict) -> dict:
    """Ensure key directory paths exist on disk."""
    root = Path(cfg["_project_root"])
    dirs_to_ensure: List[str] = []

    # Gather all path-like values
    for key in ("saved_dir", "plots_dir", "reports_dir", "upload_dir", "splits_dir"):
        _find_and_collect(cfg, key, dirs_to_ensure)

    for rel_dir in dirs_to_ensure:
        abs_dir = root / rel_dir
        abs_dir.mkdir(parents=True, exist_ok=True)

    return cfg


def _find_and_collect(d: Any, target_key: str, collector: List[str]) -> None:
    if isinstance(d, dict):
        for k, v in d.items():
            if k == target_key and isinstance(v, str):
                collector.append(v)
            else:
                _find_and_collect(v, target_key, collector)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_config: Optional[_AttrDict] = None


def get_config() -> _AttrDict:
    """Return the cached project config (loaded once on first call)."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    cfg = get_config()
    print(json.dumps(dict(cfg), indent=2, default=str))
