"""Point-in-time integrity helpers (INV-1)."""

from __future__ import annotations

import inspect
from datetime import datetime
from typing import Any, Callable

import pandas as pd


def pit_filter(df: pd.DataFrame, as_of: datetime, col: str = "observed_at") -> pd.DataFrame:
    """Return only rows with observed_at <= as_of."""
    if df.empty:
        return df
    if col not in df.columns:
        raise ValueError(f"PIT filter requires column '{col}'")
    ts = pd.to_datetime(df[col], utc=True)
    cutoff = pd.Timestamp(as_of)
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    else:
        cutoff = cutoff.tz_convert("UTC")
    return df.loc[ts <= cutoff].copy()


def assert_as_of_param(fn: Callable[..., Any], name: str | None = None) -> None:
    """Raise if a public reader lacks a required `as_of` parameter."""
    sig = inspect.signature(fn)
    if "as_of" not in sig.parameters:
        raise AssertionError(
            f"{name or fn.__qualname__} is a public store reader but lacks required `as_of` param"
        )
    param = sig.parameters["as_of"]
    if param.default is not inspect.Parameter.empty:
        raise AssertionError(
            f"{name or fn.__qualname__}: `as_of` must be required (no default)"
        )


def list_public_readers(module: Any) -> list[tuple[str, Callable[..., Any]]]:
    """Public readers on store classes defined in fund.store.* only."""
    readers: list[tuple[str, Callable[..., Any]]] = []
    for attr_name in dir(module):
        if attr_name.startswith("_"):
            continue
        obj = getattr(module, attr_name)
        if not isinstance(obj, type):
            continue
        mod_name = getattr(obj, "__module__", "") or ""
        if not mod_name.startswith("fund.store"):
            continue
        for meth_name, meth in vars(obj).items():
            if meth_name.startswith("_"):
                continue
            if not any(meth_name.startswith(p) for p in ("get_", "load_", "read_", "fetch_")):
                continue
            if callable(meth):
                readers.append((f"{obj.__name__}.{meth_name}", meth))
    return readers
