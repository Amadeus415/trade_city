"""Lightweight webhook alerts — no notification service to maintain."""

from __future__ import annotations

from typing import Any

import httpx

from fund.logging_setup import get_logger

log = get_logger(__name__)


def send_alert(
    webhook_url: str | None,
    title: str,
    body: str,
    *,
    severity: str = "warning",
    extra: dict[str, Any] | None = None,
) -> bool:
    """POST a simple JSON payload. No-ops if webhook_url is empty."""
    if not webhook_url:
        log.info("alert_skipped_no_webhook", title=title, severity=severity)
        return False
    payload = {
        "title": title,
        "body": body,
        "severity": severity,
        **(extra or {}),
    }
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(webhook_url, json=payload)
            resp.raise_for_status()
        log.info("alert_sent", title=title, severity=severity)
        return True
    except Exception as e:
        log.error("alert_failed", title=title, error=str(e))
        return False
