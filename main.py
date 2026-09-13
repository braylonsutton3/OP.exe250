"""OP.exe — one-page, conservative multi-timeframe market analysis dashboard.

Research/decision-support only. This program does not place orders and no signal,
entry, stop, or target is guaranteed. Yahoo Finance futures are continuous-contract
proxies and can differ from TopstepX/ProjectX prices.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


st.set_page_config(page_title="OP.exe", page_icon="📈", layout="wide")

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

TIMEFRAMES = {
    "1m": ("7d", "1m", None),
    "5m": ("60d", "5m", None),
    "15m": ("60d", "15m", None),
    "30m": ("60d", "30m", None),
    "4H": ("730d", "1h", "4h"),
    "1D": ("2y", "1d", None),
    "1W": ("5y", "1wk", None),
}

ALIASES = {
    "MNQ": "MNQ=F", "NQ": "NQ=F", "MES": "MES=F", "ES": "ES=F",
    "M2K": "M2K=F", "RTY": "RTY=F", "MYM": "MYM=F", "YM": "YM=F",
    "MGC": "MGC=F", "GC": "GC=F", "MCL": "MCL=F", "CL": "CL=F",
    "SIL": "SIL=F", "SI": "SI=F", "HG": "HG=F", "NG": "NG=F",
    "ZB": "ZB=F", "ZN": "ZN=F", "ZF": "ZF=F", "ZT": "ZT=F",
    "6E": "6E=F", "6B": "6B=F", "6J": "6J=F", "6A": "6A=F",
    "SPX": "^SPX", "VIX": "^VIX", "DOW": "^DJI",
}

# Dollar value of a one-point move for one contract/share.
POINT_VALUES = {
    "MNQ=F": 2.0, "NQ=F": 20.0, "MES=F": 5.0, "ES=F": 50.0,
    "M2K=F": 5.0, "RTY=F": 50.0, "MYM=F": 0.50, "YM=F": 5.0,
    "MGC=F": 10.0, "GC=F": 100.0, "MCL=F": 100.0, "CL=F": 1000.0,
    "SIL=F": 1000.0, "SI=F": 5000.0, "HG=F": 25000.0,
    "NG=F": 10000.0, "ZB=F": 1000.0, "ZN=F": 1000.0,
}

TICK_SIZES = {
    "MNQ=F": .25, "NQ=F": .25, "MES=F": .25, "ES=F": .25,
    "M2K=F": .10, "RTY=F": .10, "MYM=F": 1.0, "YM=F": 1.0,
    "MGC=F": .10, "GC=F": .10, "MCL=F": .01, "CL=F": .01,
}

POSITIVE_WORDS = {
    "beat", "beats", "upgrade", "upgraded", "growth", "surge", "record",
    "approval", "approved", "partnership", "profit", "raises", "raised",
    "bullish", "outperform", "buyback", "dividend", "wins", "strong",
}
NEGATIVE_WORDS = {
    "miss", "misses", "downgrade", "downgraded", "cuts", "cut", "loss",
    "lawsuit", "probe", "investigation", "recall", "warning", "weak",
    "bearish", "underperform", "bankruptcy", "fraud", "decline", "falls",
}


@dataclass
class Zone:
    kind: str
    lower: float
    upper: float
    midpoint: float
    created: pd.Timestamp
    index_position: int


@dataclass
class Analysis:
    timeframe: str
    signal: str
    score: float
    confidence: int
    reasons: List[str]
    atr: float
    open_fvgs: List[Zone]


def resolve_symbol(raw: str) -> str:
    value = raw.strip().upper().replace(" ", "")
    # Friendly handling of dated futures entries such as ESU26/M2KU26.
    if value in ALIASES:
        return ALIASES[value]
    for root in sorted(ALIASES, key=len, reverse=True):
        if value.startswith(root) and len(value) > len(root) and value[len(root)] in "FGHJKMNQUVXZ":
            return ALIASES[root]
    return value


def asset_class(symbol: str) -> str:
    if symbol.endswith("=F"):
        return "futures"
    if symbol.endswith("-USD"):
        return "crypto"
    if symbol.endswith("=X"):
        return "forex"
    return "equity"


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    wanted = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    out = df[wanted].copy()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in out:
            out[col] = 0.0 if col == "Volume" else np.nan
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["Open", "High", "Low", "Close"])
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


@st.cache_data(ttl=45, show_spinner=False)
def fetch_timeframes(symbol: str) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    frames: Dict[str, pd.DataFrame] = {}
    errors: List[str] = []
    ticker = yf.Ticker(symbol)
    for label, (period, interval, resample_rule) in TIMEFRAMES.items():
        try:
            raw = ticker.history(period=period, interval=interval, auto_adjust=False, prepost=False)
            raw = clean_ohlcv(raw)
            if resample_rule and not raw.empty:
                raw = raw.resample(resample_rule).agg({
                    "Open": "first", "High": "max", "Low": "min",
                    "Close": "last", "Volume": "sum",
                }).dropna(subset=["Open", "High", "Low", "Close"])
            if len(raw) < 25:
                errors.append(f"{label}: insufficient bars")
            else:
                frames[label] = raw
        except Exception as exc:
            errors.append(f"{label}: {type(exc).__name__}")
    return frames, errors


def market_status(symbol: str, one_minute: Optional[pd.DataFrame]) -> Tuple[bool, str, Optional[datetime]]:
    """Conservative session + freshness gate. A stale feed can never produce a trade signal."""
    now = datetime.now(UTC)
    now_et = now.astimezone(ET)
    kind = asset_class(symbol)
    clock_open = False
    reason = "Market closed"

    if kind == "crypto":
        clock_open = True
    elif kind == "futures":
        weekday, minute = now_et.weekday(), now_et.hour * 60 + now_et.minute
        # Approximate CME Globex: Sun 18:00–Fri 17:00 ET; daily 17:00–18:00 break.
        if weekday == 6:
            clock_open = minute >= 18 * 60
        elif weekday in (0, 1, 2, 3):
            clock_open = not (17 * 60 <= minute < 18 * 60)
        elif weekday == 4:
            clock_open = minute < 17 * 60
        reason = "Market closed (CME session/maintenance break)"
    elif kind == "forex":
        weekday, minute = now_et.weekday(), now_et.hour * 60 + now_et.minute
        clock_open = (weekday < 4) or (weekday == 4 and minute < 17 * 60) or (weekday == 6 and minute >= 17 * 60)
        reason = "Market closed (forex weekend)"
    else:
        try:
            schedule = mcal.get_calendar("NYSE").schedule(
                start_date=(now_et.date() - timedelta(days=1)),
                end_date=(now_et.date() + timedelta(days=1)),
            )
            clock_open = any(row.market_open.to_pydatetime() <= now <= row.market_close.to_pydatetime()
                             for _, row in schedule.iterrows())
            reason = "Market closed (regular NYSE session)"
        except Exception:
            minute = now_et.hour * 60 + now_et.minute
            clock_open = now_et.weekday() < 5 and 9 * 60 + 30 <= minute < 16 * 60

    if not clock_open:
        return False, reason, None
    if one_minute is None or one_minute.empty:
        return False, "Live-enough 1-minute data unavailable", None

    stamp = one_minute.index[-1]
    stamp = pd.Timestamp(stamp)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize(UTC)
    stamp_utc = stamp.tz_convert(UTC).to_pydatetime()
    age_minutes = max(0.0, (now - stamp_utc).total_seconds() / 60)
    # Yahoo can be delayed. Forty minutes is a hard safety ceiling, not proof of live data.
    if age_minutes > 40:
        return False, f"Market may be open, but data is stale ({age_minutes:.0f} minutes old)", stamp_utc
    return True, "Market open; data passed freshness check", stamp_utc


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    prev = df["Close"].shift(1)
    tr = pd.concat([(df["High"] - df["Low"]), (df["High"] - prev).abs(),
                    (df["Low"] - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / length, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / length, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def find_open_fvgs(df: pd.DataFrame, lookback: int = 220) -> List[Zone]:
    zones: List[Zone] = []
    start = max(2, len(df) - lookback)
    for i in range(start, len(df)):
        # Wick-based three-candle imbalance; stricter than body-only evidence.
        if float(df["Low"].iloc[i]) > float(df["High"].iloc[i - 2]):
            lower, upper, kind = float(df["High"].iloc[i - 2]), float(df["Low"].iloc[i]), "bullish FVG"
        elif float(df["High"].iloc[i]) < float(df["Low"].iloc[i - 2]):
            lower, upper, kind = float(df["High"].iloc[i]), float(df["Low"].iloc[i - 2]), "bearish FVG"
        else:
            continue
        later = df.iloc[i + 1:]
        fully_filled = (not later.empty) and (
            (later["Low"] <= lower).any() if kind.startswith("bullish") else (later["High"] >= upper).any()
        )
        if not fully_filled:
            zones.append(Zone(kind, lower, upper, (lower + upper) / 2, df.index[i], i))
    return zones


def recent_ifvg_direction(df: pd.DataFrame, bars: int = 10) -> int:
    """+1 when a bearish FVG recently inverted bullish; -1 for the inverse."""
    start = max(2, len(df) - 80)
    bullish_breaks: List[int] = []
    bearish_breaks: List[int] = []
    for i in range(start, len(df) - 1):
        if df["High"].iloc[i] < df["Low"].iloc[i - 2]:  # bearish gap
            upper = float(df["Low"].iloc[i - 2])
            hits = np.where(df["Close"].iloc[i + 1:].to_numpy() > upper)[0]
            if len(hits):
                bullish_breaks.append(i + 1 + int(hits[0]))
        elif df["Low"].iloc[i] > df["High"].iloc[i - 2]:  # bullish gap
            lower = float(df["High"].iloc[i - 2])
            hits = np.where(df["Close"].iloc[i + 1:].to_numpy() < lower)[0]
            if len(hits):
                bearish_breaks.append(i + 1 + int(hits[0]))
    cutoff = len(df) - bars
    bull = max(bullish_breaks, default=-1)
    bear = max(bearish_breaks, default=-1)
    if bull >= cutoff and bull > bear:
        return 1
    if bear >= cutoff and bear > bull:
        return -1
    return 0


def two_candle_confirmation(df: pd.DataFrame) -> int:
    if len(df) < 2:
        return 0
    last = df.iloc[-2:]
    if (last["Close"] > last["Open"]).all():
        return 1
    if (last["Close"] < last["Open"]).all():
        return -1
    return 0


def directional_one_minute_imbalance(df: pd.DataFrame, direction: int) -> Tuple[bool, str]:
    recent = df.iloc[-12:]
    zones = find_open_fvgs(recent, lookback=12)
    wanted = "bullish" if direction == 1 else "bearish"
    fvg_ok = any(z.kind.startswith(wanted) for z in zones)
    ifvg_ok = recent_ifvg_direction(df, bars=10) == direction
    if fvg_ok and ifvg_ok:
        return True, f"1m {wanted} FVG + IFVG"
    if fvg_ok:
        return True, f"1m {wanted} FVG"
    if ifvg_ok:
        return True, f"1m {wanted} IFVG"
    return False, "Required 1m FVG/IFVG is absent"


def fvg_retracement_setup(df15: pd.DataFrame, df1: pd.DataFrame) -> dict:
    """User's MNQ setup: open 15m target, displacement away, then 1m retracement confirmation."""
    price = float(df1["Close"].iloc[-1])
    zones = find_open_fvgs(df15)
    above = sorted([z for z in zones if z.lower > price], key=lambda z: z.lower - price)
    below = sorted([z for z in zones if z.upper < price], key=lambda z: price - z.upper)
    a15 = float(atr(df15).iloc[-1])
    candidates = []

    if above:
        z = above[0]
        after = df15.iloc[z.index_position + 1:]
        moved = not after.empty and float(after["Low"].min()) <= z.lower - .35 * a15
        away_confirm = any(
            (after["Close"].iloc[j - 1] < after["Open"].iloc[j - 1]) and
            (after["Close"].iloc[j] < after["Open"].iloc[j])
            for j in range(1, len(after))
        ) if len(after) >= 2 else False
        candles = two_candle_confirmation(df1) == 1
        imbalance, imbalance_reason = directional_one_minute_imbalance(df1, 1)
        valid = moved and away_confirm and candles and imbalance
        candidates.append({"direction": "LONG", "zone": z, "valid": valid,
                           "moved": moved, "away": away_confirm, "candles": candles,
                           "imbalance": imbalance, "imbalance_reason": imbalance_reason,
                           "distance": z.lower - price})

    if below:
        z = below[0]
        after = df15.iloc[z.index_position + 1:]
        moved = not after.empty and float(after["High"].max()) >= z.upper + .35 * a15
        away_confirm = any(
            (after["Close"].iloc[j - 1] > after["Open"].iloc[j - 1]) and
            (after["Close"].iloc[j] > after["Open"].iloc[j])
            for j in range(1, len(after))
        ) if len(after) >= 2 else False
        candles = two_candle_confirmation(df1) == -1
        imbalance, imbalance_reason = directional_one_minute_imbalance(df1, -1)
        valid = moved and away_confirm and candles and imbalance
        candidates.append({"direction": "SHORT", "zone": z, "valid": valid,
                           "moved": moved, "away": away_confirm, "candles": candles,
                           "imbalance": imbalance, "imbalance_reason": imbalance_reason,
                           "distance": price - z.upper})

    valid = [c for c in candidates if c["valid"]]
    chosen = min(valid or candidates, key=lambda c: c["distance"], default=None)
    if chosen is None:
        return {"signal": "WAIT", "reason": "No open 15m FVG target exists", "zone": None}
    if not chosen["valid"]:
        missing = []
        if not chosen["moved"]: missing.append("clear displacement away")
        if not chosen["away"]: missing.append("historical two-candle move-away confirmation")
        if not chosen["candles"]: missing.append("two current 1m candles toward target")
        if not chosen["imbalance"]: missing.append(chosen["imbalance_reason"])
        return {"signal": "WAIT", "reason": "Missing: " + "; ".join(missing), "zone": chosen["zone"],
                "candidate": chosen["direction"]}
    return {"signal": chosen["direction"], "reason": "Complete 15m→1m FVG retracement setup",
            "zone": chosen["zone"], "candidate": chosen["direction"]}


@st.cache_data(ttl=300, show_spinner=False)
def headline_catalyst(symbol: str) -> Tuple[float, List[str]]:
    """Small, capped headline-keyword input. Never decisive by itself."""
    try:
        news = yf.Ticker(symbol).news or []
    except Exception:
        return 0.0, []
    titles: List[str] = []
    raw_score = 0
    for item in news[:8]:
        content = item.get("content", item) if isinstance(item, dict) else {}
        title = str(content.get("title", "")).strip()
        if not title:
            continue
        titles.append(title)
        words = set(title.lower().replace(":", " ").replace(",", " ").split())
        raw_score += len(words & POSITIVE_WORDS) - len(words & NEGATIVE_WORDS)
    return float(np.clip(raw_score / 3, -1, 1)), titles[:5]


def analyze_timeframe(label: str, df: pd.DataFrame, catalyst: float, threshold: float) -> Analysis:
    d = df.copy()
    close = d["Close"]
    d["SMA20"] = close.rolling(20).mean()
    d["SMA50"] = close.rolling(50).mean()
    d["RSI"] = rsi(close)
    d["ATR"] = atr(d)
    d["EMA12"] = close.ewm(span=12, adjust=False).mean()
    d["EMA26"] = close.ewm(span=26, adjust=False).mean()
    d["MACD_H"] = (d["EMA12"] - d["EMA26"]) - (d["EMA12"] - d["EMA26"]).ewm(span=9, adjust=False).mean()
    d["VOL_AVG"] = d["Volume"].rolling(20).mean()
    last = d.iloc[-1]
    price = float(last["Close"])
    categories: Dict[str, float] = {}
    reasons: List[str] = []

    # Trend category (SMA relationships are intentionally not counted as separate categories).
    trend = 0.0
    if pd.notna(last["SMA20"]) and pd.notna(last["SMA50"]):
        if price > last["SMA20"] > last["SMA50"]:
            trend = 1.0; reasons.append("price > SMA20 > SMA50")
        elif price < last["SMA20"] < last["SMA50"]:
            trend = -1.0; reasons.append("price < SMA20 < SMA50")
        elif price > last["SMA20"]:
            trend = .35; reasons.append("price above SMA20; trend mixed")
        elif price < last["SMA20"]:
            trend = -.35; reasons.append("price below SMA20; trend mixed")
    categories["trend"] = trend * 1.35

    # Structure: recent BOS plus directional swing progression.
    prior_high = d["High"].shift(1).rolling(20).max()
    prior_low = d["Low"].shift(1).rolling(20).min()
    bull_bos = bool((d["Close"].iloc[-5:] > prior_high.iloc[-5:]).any())
    bear_bos = bool((d["Close"].iloc[-5:] < prior_low.iloc[-5:]).any())
    swing_delta = price - float(close.iloc[-10]) if len(d) >= 10 else 0
    if bull_bos and not bear_bos:
        categories["structure"] = 1.5; reasons.append("bullish break of structure (BOS)")
    elif bear_bos and not bull_bos:
        categories["structure"] = -1.5; reasons.append("bearish break of structure (BOS)")
    else:
        categories["structure"] = .45 * np.sign(swing_delta)
        reasons.append("structure mixed; using minor swing progression")

    # Liquidity sweep category.
    ph = float(d["High"].iloc[-22:-2].max()) if len(d) >= 22 else float(d["High"].iloc[:-2].max())
    pl = float(d["Low"].iloc[-22:-2].min()) if len(d) >= 22 else float(d["Low"].iloc[:-2].min())
    recent = d.iloc[-2:]
    sell_sweep = bool((recent["Low"] < pl).any() and price > pl)
    buy_sweep = bool((recent["High"] > ph).any() and price < ph)
    if sell_sweep and not buy_sweep:
        categories["liquidity"] = .95; reasons.append("sell-side liquidity sweep/reclaim")
    elif buy_sweep and not sell_sweep:
        categories["liquidity"] = -.95; reasons.append("buy-side liquidity sweep/rejection")
    else:
        categories["liquidity"] = 0.0

    # Momentum category: RSI and MACD are combined to avoid double counting.
    mom = 0.0
    if last["RSI"] >= 55 and last["MACD_H"] > 0: mom = 1.0
    elif last["RSI"] <= 45 and last["MACD_H"] < 0: mom = -1.0
    elif last["MACD_H"] > 0: mom = .35
    elif last["MACD_H"] < 0: mom = -.35
    categories["momentum"] = mom
    reasons.append(f"RSI {last['RSI']:.0f}; MACD histogram {'positive' if last['MACD_H'] > 0 else 'negative'}")

    # Volume only strengthens the candle's direction; zero-volume symbols are neutral.
    volume_ratio = float(last["Volume"] / last["VOL_AVG"]) if pd.notna(last["VOL_AVG"]) and last["VOL_AVG"] > 0 else 0
    candle_dir = np.sign(float(last["Close"] - last["Open"]))
    categories["volume"] = float(candle_dir * min(.8, max(0, volume_ratio - 1)))
    if volume_ratio >= 1.25:
        reasons.append(f"volume expansion {volume_ratio:.1f}× average")

    # Nearby open imbalances; one bounded category regardless of number of zones.
    zones = find_open_fvgs(d)
    above = [z for z in zones if z.lower > price]
    below = [z for z in zones if z.upper < price]
    if above and not below:
        categories["imbalance"] = .6; reasons.append("open FVG liquidity target above")
    elif below and not above:
        categories["imbalance"] = -.6; reasons.append("open FVG liquidity target below")
    else:
        categories["imbalance"] = 0.0

    categories["catalyst"] = catalyst * .35
    if catalyst > .15: reasons.append("recent headline catalyst tilt positive")
    elif catalyst < -.15: reasons.append("recent headline catalyst tilt negative")

    score = float(sum(categories.values()))
    max_score = 6.55
    confidence = int(np.clip(round(abs(score) / max_score * 100), 0, 99))
    signal = "LONG" if score >= threshold else "SHORT" if score <= -threshold else "WAIT"
    if signal == "WAIT": reasons.insert(0, "evidence is weak or conflicting")
    return Analysis(label, signal, score, confidence, reasons, float(last["ATR"]), zones)


def combined_signal(analyses: Dict[str, Analysis], setup: dict, symbol: str) -> Tuple[str, str]:
    weights = {"1m": 1.1, "5m": 1.25, "15m": 1.55, "30m": 1.45, "4H": 1.1, "1D": .7, "1W": .4}
    directional = sum(np.sign(a.score) * min(abs(a.score), 4) * weights.get(k, 1) for k, a in analyses.items())
    intraday = [analyses[k].signal for k in ("1m", "5m", "15m", "30m") if k in analyses]
    if symbol == "MNQ=F":
        # The user's latest MNQ strategy is a hard gate, not merely another indicator.
        if setup.get("signal") not in ("LONG", "SHORT"):
            return "WAIT", "MNQ 15m→1m retracement confirmation is incomplete"
        desired = setup["signal"]
        opposing = "SHORT" if desired == "LONG" else "LONG"
        support = intraday.count(desired)
        if intraday.count(opposing) > 0 or support < 2:
            return "WAIT", "FVG setup exists, but broader intraday confluence does not agree"
        return desired, "Complete MNQ FVG setup agrees with multi-timeframe confluence"
    if directional >= 7 and intraday.count("SHORT") == 0:
        return "LONG", "weighted multi-timeframe evidence is bullish"
    if directional <= -7 and intraday.count("LONG") == 0:
        return "SHORT", "weighted multi-timeframe evidence is bearish"
    return "WAIT", "multi-timeframe evidence is weak or conflicting"


def round_tick(value: float, tick: float) -> float:
    return round(round(value / tick) * tick, 8)


def trade_plan(signal: str, price: float, df1: pd.DataFrame, analysis15: Analysis,
               setup: dict, symbol: str, contracts: int, max_risk: float) -> dict:
    if signal not in ("LONG", "SHORT"):
        return {}
    point_value = POINT_VALUES.get(symbol, 1.0)
    tick = TICK_SIZES.get(symbol, .01 if price < 1000 else .25)
    cap_distance = max_risk / max(contracts * point_value, .0001)
    a1 = max(float(atr(df1).iloc[-1]), tick)
    recent_low = float(df1["Low"].iloc[-12:].min())
    recent_high = float(df1["High"].iloc[-12:].max())
    entry = round_tick(price, tick)
    zone = setup.get("zone")

    if signal == "LONG":
        structural = min(recent_low - tick, entry - .75 * a1)
        stop = max(structural, entry - cap_distance)
        nearby = sorted([z for z in analysis15.open_fvgs if z.lower > entry], key=lambda z: z.lower)
        chosen = zone if zone is not None and zone.lower > entry else (nearby[0] if nearby else None)
        tp1 = chosen.midpoint if chosen else entry + 1.5 * (entry - stop)
        tp2 = chosen.upper if chosen else entry + 2.0 * (entry - stop)
    else:
        structural = max(recent_high + tick, entry + .75 * a1)
        stop = min(structural, entry + cap_distance)
        nearby = sorted([z for z in analysis15.open_fvgs if z.upper < entry], key=lambda z: -z.upper)
        chosen = zone if zone is not None and zone.upper < entry else (nearby[0] if nearby else None)
        tp1 = chosen.midpoint if chosen else entry - 1.5 * (stop - entry)
        tp2 = chosen.lower if chosen else entry - 2.0 * (stop - entry)

    stop, tp1, tp2 = (round_tick(x, tick) for x in (stop, tp1, tp2))
    risk = abs(entry - stop) * point_value * contracts
    reward = abs(tp2 - entry) * point_value * contracts
    return {
        "entry": entry, "stop": stop, "tp1": tp1, "tp2": tp2,
        "risk": risk, "reward": reward, "reward_per_contract": reward / contracts,
        "rr": reward / risk if risk else 0, "point_value": point_value,
        "target_basis": "15m FVG midpoint and far edge" if chosen else "ATR/structure fallback",
    }


def chart(df: pd.DataFrame, setup: dict, title: str) -> go.Figure:
    view = df.iloc[-120:]
    fig = go.Figure(go.Candlestick(x=view.index, open=view["Open"], high=view["High"],
                                   low=view["Low"], close=view["Close"], name="Price"))
    z = setup.get("zone")
    if z is not None:
        color = "rgba(46, 204, 113, .18)" if setup.get("candidate") == "LONG" else "rgba(231, 76, 60, .18)"
        fig.add_hrect(y0=z.lower, y1=z.upper, fillcolor=color, line_width=1)
        fig.add_hline(y=z.midpoint, line_dash="dot", line_color="#f1c40f")
    fig.update_layout(title=title, height=430, xaxis_rangeslider_visible=False,
                      margin=dict(l=10, r=10, t=45, b=10), template="plotly_dark")
    return fig


st.title("OP.exe")
st.caption("Conservative market-structure decision support • no automatic trading • no guaranteed outcomes")

with st.form("controls"):
    c1, c2, c3, c4, c5 = st.columns([2.2, 1, 1.2, 1.3, 1.4])
    raw_symbol = c1.text_input("Ticker", value="MNQ", help="Examples: MNQ, MES, MGC, SPY, AAPL, BTC-USD")
    contracts = int(c2.number_input("Contracts/shares", min_value=1, max_value=1000, value=3, step=1))
    max_risk = float(c3.number_input("Maximum total risk ($)", min_value=1.0, value=300.0, step=25.0))
    strictness = c4.selectbox("Signal filter", ["Strict", "Balanced"], index=0)
    minimum_reward = float(c5.number_input("Minimum target value/contract ($)", min_value=0.0, value=100.0, step=25.0))
    run = st.form_submit_button("Analyze", use_container_width=True)

if run or raw_symbol:
    symbol = resolve_symbol(raw_symbol)
    threshold = 3.15 if strictness == "Strict" else 2.55
    with st.spinner(f"Analyzing {symbol}…"):
        frames, data_errors = fetch_timeframes(symbol)

    if not frames:
        st.error("No usable price data was returned. Check the ticker and try again.")
        st.stop()
        raise SystemExit(1)  # Also stops cleanly if somebody runs `python main.py` by mistake.

    df1 = frames.get("1m")
    is_open, market_reason, last_stamp = market_status(symbol, df1)
    price_frame = df1 if df1 is not None else next(iter(frames.values()))
    current_price = float(price_frame["Close"].iloc[-1])
    catalyst_symbol = "QQQ" if symbol in ("MNQ=F", "NQ=F") else "SPY" if symbol in ("MES=F", "ES=F") else symbol
    catalyst, headlines = headline_catalyst(catalyst_symbol)
    analyses = {k: analyze_timeframe(k, df, catalyst, threshold) for k, df in frames.items()}

    if "15m" in frames and df1 is not None:
        setup = fvg_retracement_setup(frames["15m"], df1)
    else:
        setup = {"signal": "WAIT", "reason": "15m or 1m data unavailable", "zone": None}

    raw_signal, signal_reason = combined_signal(analyses, setup, symbol)
    final_signal = raw_signal if is_open else "WAIT"
    final_reason = signal_reason if is_open else market_reason

    plan = {}
    if final_signal in ("LONG", "SHORT") and df1 is not None and "15m" in analyses:
        plan = trade_plan(final_signal, current_price, df1, analyses["15m"], setup,
                          symbol, contracts, max_risk)
        if plan and plan["reward_per_contract"] < minimum_reward:
            final_signal = "WAIT"
            final_reason = f"Projected target value is under ${minimum_reward:.0f} per contract"
            plan = {}
        elif plan and plan["rr"] < 1.0:
            final_signal = "WAIT"
            final_reason = "Available structural target does not provide at least 1:1 reward:risk"
            plan = {}

    signal_color = {"LONG": "#21c55d", "SHORT": "#ef4444", "WAIT": "#f59e0b"}[final_signal]
    st.markdown(
        f"<div style='padding:18px;border-radius:14px;background:#111827;border-left:8px solid {signal_color}'>"
        f"<div style='font-size:34px;font-weight:800;color:{signal_color}'>{final_signal}</div>"
        f"<div style='font-size:17px'>{final_reason}</div></div>", unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Symbol used", symbol)
    m2.metric("Current proxy price", f"{current_price:,.2f}")
    m3.metric("Market/data gate", "PASS" if is_open else "WAIT")
    m4.metric("Latest 1m bar", last_stamp.astimezone(ET).strftime("%I:%M %p ET") if last_stamp else "Unavailable")

    if plan:
        st.subheader("Trade plan")
        p1, p2, p3, p4, p5, p6 = st.columns(6)
        p1.metric("Entry", f"{plan['entry']:,.2f}")
        p2.metric("Stop loss", f"{plan['stop']:,.2f}")
        p3.metric("TP1 · midpoint", f"{plan['tp1']:,.2f}")
        p4.metric("TP2 · far edge", f"{plan['tp2']:,.2f}")
        p5.metric("Planned risk", f"${plan['risk']:,.0f}")
        p6.metric("Reward:risk", f"{plan['rr']:.2f}:1")
        st.caption(f"Target basis: {plan['target_basis']}. Stop is structural but capped so planned total risk does not exceed ${max_risk:,.0f} before slippage/fees.")

    st.subheader("Your MNQ 15m → 1m setup")
    z = setup.get("zone")
    s1, s2, s3 = st.columns([1, 2.4, 1.6])
    s1.metric("Setup", setup.get("signal", "WAIT"))
    s2.write(setup.get("reason", "Unavailable"))
    if z:
        s3.write(f"15m FVG: **{z.lower:,.2f}–{z.upper:,.2f}**  \\n+Midpoint: **{z.midpoint:,.2f}**")
    else:
        s3.write("No active target zone")

    st.subheader("Multi-timeframe evidence")
    rows = []
    for label in TIMEFRAMES:
        if label not in analyses:
            rows.append({"Timeframe": label, "Signal": "NO DATA", "Score": "—", "Confidence": "—", "Top evidence": "Unavailable"})
            continue
        a = analyses[label]
        rows.append({"Timeframe": label, "Signal": a.signal, "Score": f"{a.score:+.2f}",
                     "Confidence": f"{a.confidence}%", "Top evidence": "; ".join(a.reasons[:3])})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    if "15m" in frames:
        st.plotly_chart(chart(frames["15m"], setup, f"{symbol} · 15-minute FVG context"), use_container_width=True)

    with st.expander("Full evidence, catalysts, and data notes"):
        for label, a in analyses.items():
            st.markdown(f"**{label} — {a.signal} ({a.score:+.2f})**")
            st.write(" • ".join(a.reasons))
        st.markdown("**Recent catalyst headlines (small, capped score)**")
        if headlines:
            for h in headlines:
                st.write(f"• {h}")
        else:
            st.write("No usable recent headlines returned; catalyst score stayed neutral.")
        if data_errors:
            st.warning("Data notes: " + " | ".join(data_errors))

    if asset_class(symbol) == "futures":
        st.warning("Yahoo Finance futures data is a continuous-contract proxy and may be delayed or differ from the exact TopstepX/ProjectX contract. Confirm entry, stop, target, and session status on your broker before acting.")
    st.info("Accuracy is the objective, not a guarantee. OP.exe intentionally returns WAIT when confirmation, reward, market status, freshness, or confluence is inadequate.")
