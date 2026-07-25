"""Masking modes + mock agent schema validity."""

from __future__ import annotations

from fund.agent.masking import MaskingContext
from fund.agent.pm import PortfolioManager
from fund.config import AgentConfig
from fund.types import MaskingMode


def test_stock_blind_consistent():
    ctx = MaskingContext(mode=MaskingMode.STOCK_BLIND)
    p = "UNIVERSE SNAPSHOT\nAAPL  +8.2%\nMSFT  -1.0%\nAAPL again"
    masked = ctx.mask_packet(p)
    assert "AAPL" not in masked
    assert "MSFT" not in masked
    assert "SYM_" in masked
    # consistent alias
    a1 = ctx.alias_symbol("AAPL")
    a2 = ctx.alias_symbol("AAPL")
    assert a1 == a2
    assert ctx.unmask_symbol(a1) == "AAPL"


def test_blinded_masks_dates():
    ctx = MaskingContext(mode=MaskingMode.BLINDED)
    p = "as_of 2024-06-03 16:15 ET\nNVDA mom"
    masked = ctx.mask_packet(p)
    assert "2024-06-03" not in masked
    assert "NVDA" not in masked


def test_mock_pm_schema():
    cfg = AgentConfig(provider="mock", cache_enabled=False)
    pm = PortfolioManager(cfg, cache=None)
    packet = """UNIVERSE SNAPSHOT — as_of 2024-06-03 16:15 ET
sym   mom_63d
AAPL      +8.2%
MSFT      -1.0%
GOOGL     +3.0%

PORTFOLIO — equity $10,000.00  cash $10,000.00  gross 0.0%
sym  weight  days_held  cluster

CONSTRAINTS max_position 0.10
RISK STATE: NORMAL
"""
    proposals, meta = pm.run(packet)
    assert "prompt_version" in meta
    for p in proposals:
        assert p.thesis
        assert p.invalidation
        assert 0 <= float(p.target_weight) <= 1
