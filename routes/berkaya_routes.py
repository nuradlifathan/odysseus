"""Berkaya intelligence endpoint.

GET /api/berkaya/context?pair=BTCUSDT&direction=LONG

Returns macro context (Fear & Greed, regime, news sentiment) for Berkaya
trading bot to enrich signal confidence before order approval.

No auth required — local-only endpoint (Odysseus binds 127.0.0.1).
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import httpx
from fastapi import APIRouter
from pydantic import BaseModel

from core.database import SessionLocal, ModelEndpoint
from src.llm_core import llm_call_async
from src.settings import load_settings

logger = logging.getLogger(__name__)

_FNG_URL = "https://api.alternative.me/fng/?limit=1"
_FNG_TIMEOUT = 5.0
_LLM_TIMEOUT = 20

# Simple in-process cache: (pair, direction) → (result, expires_at)
_ctx_cache: dict[tuple[str, str], tuple[dict, float]] = {}

# Daily brief cache: real news dari RSS + LLM analyze
_NEWS_FEEDS = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
]
_BRIEF_TTL = 900.0   # 15 menit
_brief_cache: dict | None = None
_brief_expires: float = 0.0
_scheduler_thread: threading.Thread | None = None


def _get_default_llm() -> tuple[str, str] | None:
    """Return (base_url, model) from Odysseus default endpoint settings."""
    settings = load_settings()
    ep_id = settings.get("default_endpoint_id", "")
    model = settings.get("default_model", "")
    if not ep_id:
        return None
    try:
        with SessionLocal() as db:
            ep = db.query(ModelEndpoint).filter(ModelEndpoint.id == ep_id).first()
            if ep and ep.base_url:
                return ep.base_url, model
    except Exception as exc:
        logger.warning("berkaya: gagal ambil endpoint dari DB: %s", exc)
    return None


_JSON_KEY_RE = re.compile(r'"([A-Za-z_]+)"\s*:')


def _fix_unclosed_array(text: str) -> str:
    """Auto-close an array when a fresh `"key":` pair starts while still
    inside it. LFM2-8B-A1B has a recurring tic on this exact schema: it
    never closes `"pros": [...]` before writing `"Cons": [...]` (note the
    capitalization slip too — handled separately by lowercasing keys).
    Also auto-closes any brackets still open at the end, which doubles as
    a safety net for genuinely truncated completions.
    """
    out: list[str] = []
    stack: list[str] = []
    in_string = False
    escape = False
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if in_string:
            out.append(ch)
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            if stack and stack[-1] == '[' and _JSON_KEY_RE.match(text, i):
                # Roll back the trailing ",\s*" we already emitted for this
                # (non-existent) next array item, close the array instead.
                j = len(out)
                while j > 0 and out[j - 1].isspace():
                    j -= 1
                had_comma = j > 0 and out[j - 1] == ','
                if had_comma:
                    j -= 1
                del out[j:]
                out.append(']')
                if had_comma:
                    out.append(',')
                stack.pop()
            in_string = True
            out.append(ch)
            i += 1
            continue
        if ch == '{':
            stack.append('{')
        elif ch == '[':
            stack.append('[')
        elif ch == '}':
            # Model closed the object without closing the array it was
            # still inside (e.g. ends after the last "pros" item with no
            # "cons" key at all) — close the array(s) first.
            while stack and stack[-1] == '[':
                j = len(out)
                while j > 0 and out[j - 1].isspace():
                    j -= 1
                if j > 0 and out[j - 1] == ',':
                    j -= 1
                del out[j:]
                out.append(']')
                stack.pop()
            if stack and stack[-1] == '{':
                stack.pop()
        elif ch == ']':
            if stack and stack[-1] == '[':
                stack.pop()
        out.append(ch)
        i += 1
    while stack:
        out.append('}' if stack.pop() == '{' else ']')
    return ''.join(out)


def _fix_missing_commas(text: str) -> str:
    return re.sub(r'([\]}"])(\s*\n?\s*)(["\[{])', r'\1,\2\3', text)


def _extract_json(raw: str) -> dict | None:
    """Parse a JSON object out of a raw LLM completion.

    Local models routinely produce near-valid JSON: wrapped in markdown
    fences, followed by trailing prose, missing a comma between array/
    object items, or (LFM2's recurring tic) leaving an array unclosed
    before the next key starts. A plain `re.search(r'\\{.*\\}')` +
    `json.loads()` fails on all of these, which empirically caused ~33-47%
    parse failures depending on model. `raw_decode` tolerates trailing
    garbage; the two repair passes below recover the rest.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        text = text.strip()

    start = text.find("{")
    if start == -1:
        return None
    text = text[start:]

    comma_fixed = _fix_missing_commas(text)
    candidates = (text, comma_fixed, _fix_unclosed_array(text), _fix_unclosed_array(comma_fixed))

    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            obj, _ = decoder.raw_decode(candidate)
            if isinstance(obj, dict):
                return {k.lower(): v for k, v in obj.items()}
        except (json.JSONDecodeError, ValueError):
            continue
    return None


async def _fetch_fear_greed() -> tuple[int, str]:
    """Fetch Fear & Greed index. Returns (value, label)."""
    try:
        async with httpx.AsyncClient(timeout=_FNG_TIMEOUT) as client:
            resp = await client.get(_FNG_URL)
            resp.raise_for_status()
            data = resp.json().get("data", [{}])[0]
            value = int(data.get("value", 50))
            label = data.get("value_classification", "Neutral")
            return value, label
    except Exception as exc:
        logger.warning("berkaya: gagal fetch F&G: %s", exc)
        return 50, "Neutral"


def _classify_regime(fg: int, direction: str) -> tuple[str, int]:
    """Return (regime, confidence_delta) based on Fear & Greed value and trade direction."""
    is_long = direction == "LONG"
    if fg < 20:
        # Extreme Fear — volatility tinggi, bahaya kedua arah
        return "CAUTION", -15
    if fg < 40:
        # Bear market — bagus untuk SHORT, buruk untuk LONG
        return "BEAR", -10 if is_long else +5
    if fg <= 60:
        return "NEUTRAL", 0
    if fg <= 80:
        # Bull market — bagus untuk LONG, buruk untuk SHORT
        return "BULL", 0 if is_long else -10
    # Extreme Greed — overbought, berbahaya untuk kedua arah
    return "CAUTION", -15


async def _classify_news(pair: str, direction: str, llm_url: str, model: str) -> tuple[str, str, int, bool]:
    """
    Call local LLM to get news sentiment for pair.
    Returns (sentiment, summary, delta, veto).
    Falls back to NEUTRAL if LLM unavailable.
    """
    prompt = f"""You are a crypto trading risk analyst. Assess the current market sentiment for {pair} futures trading.

Current trade direction being considered: {direction}

Based on your knowledge of recent crypto market conditions, answer ONLY in this exact JSON format (no markdown, no explanation):
{{"sentiment": "NEUTRAL", "summary": "Brief 1-sentence summary in English.", "veto": false}}

Rules:
- sentiment must be one of: CRITICAL, NEGATIVE, NEUTRAL, POSITIVE
- CRITICAL (veto=true): only for major black swan events (exchange hack >$50M, major regulatory ban, exchange collapse)
- NEGATIVE: bearish news — regulatory crackdown, SEC enforcement, significant fraud/scam
- POSITIVE: bullish news — ETF approval, major institutional adoption, significant positive partnership
- NEUTRAL: normal market conditions or mixed news
- summary: max 15 words, factual, direction-aware
- veto: true ONLY when sentiment=CRITICAL"""

    for attempt in range(2):
        try:
            raw = await llm_call_async(
                url=llm_url,
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=120,
                timeout=_LLM_TIMEOUT,
            )
            parsed = _extract_json(raw)
            if parsed is None:
                logger.warning("berkaya: LLM classify JSON tidak valid (attempt %d): %s", attempt + 1, raw[:200])
                continue
            sentiment = str(parsed.get("sentiment", "NEUTRAL")).upper()
            if sentiment not in ("CRITICAL", "NEGATIVE", "NEUTRAL", "POSITIVE"):
                sentiment = "NEUTRAL"
            summary = str(parsed.get("summary", "No major news."))[:200]
            veto = bool(parsed.get("veto", False)) and sentiment == "CRITICAL"
            # Delta direction-aware: POSITIVE bagus untuk LONG, buruk untuk SHORT. Sebaliknya NEGATIVE.
            is_long = direction == "LONG"
            delta_map = {
                "CRITICAL": -30,                         # veto — buruk untuk semua arah
                "NEGATIVE": -10 if is_long else +5,      # bearish news: buruk LONG, bantu SHORT
                "NEUTRAL":  0,
                "POSITIVE": +5 if is_long else -5,       # bullish news: bantu LONG, buruk SHORT
            }
            return sentiment, summary, delta_map.get(sentiment, 0), veto
        except Exception as exc:
            logger.warning("berkaya: LLM classify gagal (attempt %d): %s", attempt + 1, exc)

    return "NEUTRAL", "News classification unavailable.", 0, False


async def _fetch_headlines(max_items: int = 12) -> list[str]:
    """Fetch crypto headlines dari RSS feeds (CoinDesk + Cointelegraph)."""
    headlines: list[str] = []
    for feed_url in _NEWS_FEEDS:
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.get(feed_url, headers={"User-Agent": "Mozilla/5.0"})
                resp.raise_for_status()
                root = ET.fromstring(resp.content)
                for item in root.iter("item"):
                    title = (item.findtext("title") or "").strip()
                    if title and title not in headlines:
                        headlines.append(title)
                        if len(headlines) >= max_items:
                            return headlines
        except Exception as exc:
            logger.warning("berkaya/brief: feed %s gagal: %s", feed_url, exc)
    return headlines


async def _brief_analyze(headlines: list[str], llm_url: str, model: str) -> dict:
    """Analisis headlines dengan LLM, return sentiment + summary + veto."""
    headlines_text = "\n".join(f"- {h}" for h in headlines[:12])
    chat_url = llm_url.rstrip("/")
    if not chat_url.endswith("/chat/completions"):
        if chat_url.endswith("/v1"):
            chat_url = chat_url[:-3]
        chat_url = chat_url.rstrip("/") + "/v1/chat/completions"

    prompt = f"""You are a crypto market analyst. Analyze these latest crypto news headlines and assess the overall market sentiment for Bitcoin and Ethereum futures trading.

Headlines:
{headlines_text}

Respond ONLY with this exact JSON (no markdown, no extra text):
{{"sentiment": "NEUTRAL", "summary": "1-sentence max 20 words", "veto": false}}

Rules:
- sentiment: CRITICAL | NEGATIVE | NEUTRAL | POSITIVE
- CRITICAL (veto=true): only catastrophic events (major exchange hack/collapse, nation-state ban)
- NEGATIVE: regulatory crackdown, major fraud, bearish macro
- POSITIVE: ETF approval, major institutional adoption, positive regulatory clarity
- NEUTRAL: mixed or routine news
- summary: factual, concise, max 20 words"""

    for attempt in range(2):
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.post(
                    chat_url,
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 120,
                        "temperature": 0.1,
                    },
                )
                resp.raise_for_status()
                raw = resp.json()["choices"][0]["message"]["content"].strip()

            parsed = _extract_json(raw)
            if parsed is None:
                logger.warning("berkaya/brief: JSON tidak valid (attempt %d): %s", attempt + 1, raw[:200])
                continue
            sentiment = str(parsed.get("sentiment", "NEUTRAL")).upper()
            if sentiment not in ("CRITICAL", "NEGATIVE", "NEUTRAL", "POSITIVE"):
                sentiment = "NEUTRAL"
            veto = bool(parsed.get("veto", False)) and sentiment == "CRITICAL"
            return {
                "sentiment": sentiment,
                "summary": str(parsed.get("summary", "Mixed crypto market news."))[:200],
                "veto": veto,
            }
        except Exception as exc:
            logger.warning("berkaya/brief: LLM analyze gagal (attempt %d): %s", attempt + 1, exc)

    return {"sentiment": "NEUTRAL", "summary": "News analysis unavailable.", "veto": False}


_memory_manager = None  # injected via setup_berkaya_routes()

_router = APIRouter(prefix="/api/berkaya", tags=["berkaya"])


@_router.get("/context")
async def get_context(pair: str, direction: str, fast: bool = False, session: str | None = None) -> dict:
    pair = pair.upper()
    direction = direction.upper()
    if direction not in ("LONG", "SHORT"):
        direction = "LONG"
    cache_key = (pair, direction, session)

    if cache_key in _ctx_cache:
        cached, expires_at = _ctx_cache[cache_key]
        if time.time() < expires_at:
            logger.debug("berkaya: cache hit untuk %s %s", pair, direction)
            return cached

    fg_value, fg_label = await _fetch_fear_greed()
    regime, fg_delta = _classify_regime(fg_value, direction)

    news_sentiment = "NEUTRAL"
    news_summary = "No major news."
    news_delta = 0
    veto = False

    if not fast:
        is_long = direction == "LONG"
        delta_map_news = {
            "CRITICAL": -30,
            "NEGATIVE": -10 if is_long else +5,
            "NEUTRAL":  0,
            "POSITIVE": +5 if is_long else -5,
        }

        if _brief_cache and time.time() < _brief_expires:
            # Pakai daily brief (real news dari RSS) — lebih akurat dari training data
            news_sentiment = _brief_cache["sentiment"]
            news_summary   = _brief_cache["summary"]
            veto           = _brief_cache.get("veto", False)
            news_delta     = delta_map_news.get(news_sentiment, 0)
            logger.debug("berkaya: pakai daily brief → %s", news_sentiment)
        else:
            llm_info = _get_default_llm()
            if llm_info:
                llm_url, model = llm_info
                news_sentiment, news_summary, news_delta, veto = await _classify_news(
                    pair, direction, llm_url, model
                )
            else:
                logger.warning("berkaya: tidak ada daily brief atau LLM — skip news sentiment")

    if veto:
        regime = "BLACKOUT"
        confidence_delta = -30
    else:
        confidence_delta = fg_delta + news_delta
        if session == "OFF":
            # Di luar jam aktif utama, sinyal makro kurang reliable — redam magnitude-nya.
            confidence_delta = int(confidence_delta * 0.8)

    ttl = 120
    result = {
        "regime": regime,
        "fear_greed": fg_value,
        "fear_greed_label": fg_label,
        "news_sentiment": news_sentiment,
        "news_summary": news_summary,
        "confidence_delta": confidence_delta,
        "veto": veto,
        "session": session,
        "ttl_seconds": ttl,
        "pair": pair,
        "direction": direction,
    }

    _ctx_cache[cache_key] = (result, time.time() + ttl)
    logger.info(
        "berkaya: context %s %s session=%s → regime=%s F&G=%d sentiment=%s delta=%+d veto=%s",
        pair, direction, session or "-", regime, fg_value, news_sentiment, confidence_delta, veto,
    )
    return result


class SignalPayload(BaseModel):
    pair: str
    direction: str
    entry: float
    sl: float
    tp: float
    confidence: float
    strategy: str
    indicator_context: str = ""
    live_market_context: str = ""
    macro_section: str = ""
    memory_section: str = ""
    winrate_section: str = ""


_ANALYZE_SYSTEM = """\
Kamu adalah crypto futures trading analyst berpengalaman.
Analisis sinyal trading dan beri second opinion singkat.

SELALU respond dengan JSON valid, tidak ada teks lain di luar JSON:
{
  "verdict": "VALID" | "CAUTION" | "RISKY",
  "reasoning": "penjelasan 2-3 kalimat",
  "risks": ["risiko 1", "risiko 2"],
  "confidence_delta": <integer -30 sampai 10>
}

Kriteria verdict:
- VALID    : indikator konsisten, setup bersih, layak di-approve
- CAUTION  : ada 1-2 faktor perlu diperhatikan, masih bisa approve tapi waspada
- RISKY    : banyak faktor berlawanan arah, disarankan skip

confidence_delta: negatif jika ada kelemahan, positif jika setup sangat solid."""


@_router.post("/analyze")
async def analyze_signal(payload: SignalPayload) -> dict:
    """Analisis sinyal trading via local LLM Odysseus. Pengganti Anthropic API."""
    llm_info = _get_default_llm()
    if not llm_info:
        return {"error": "Tidak ada LLM endpoint di Odysseus — atur default endpoint di Settings Odysseus."}

    llm_url, model = llm_info
    sl_pct = abs((payload.sl - payload.entry) / payload.entry * 100) if payload.entry else 0
    tp_pct = abs((payload.tp - payload.entry) / payload.entry * 100) if payload.entry else 0
    if payload.direction == "LONG":
        rr = (payload.tp - payload.entry) / (payload.entry - payload.sl) if payload.entry != payload.sl else 0
    else:
        rr = (payload.entry - payload.tp) / (payload.sl - payload.entry) if payload.sl != payload.entry else 0

    user_msg = f"""Analisis sinyal berikut dan berikan second opinion:

SINYAL:
- Pair: {payload.pair} (Binance Futures, leverage 5x isolated)
- Direction: {payload.direction}
- Entry: {payload.entry:,.4f}
- Stop Loss: {payload.sl:,.4f} ({sl_pct:.2f}% dari entry)
- Take Profit: {payload.tp:,.4f} ({tp_pct:.2f}% dari entry)
- Risk:Reward = 1:{rr:.2f}
- Confidence scraper: {payload.confidence}%
- Strategy: {payload.strategy}
{payload.indicator_context}{payload.live_market_context}{payload.macro_section}{payload.memory_section}{payload.winrate_section}
Berikan verdict dan analisis singkat."""

    # Panggil langsung ke OpenAI-compatible endpoint (Ollama /v1/chat/completions)
    chat_url = llm_url.rstrip("/")
    if not chat_url.endswith("/chat/completions"):
        if chat_url.endswith("/v1"):
            chat_url = chat_url[:-3]  # buang /v1, tambah lagi di bawah
        chat_url = chat_url.rstrip("/") + "/v1/chat/completions"

    raw = ""
    try:
        async with httpx.AsyncClient(timeout=35.0) as client:
            for attempt in range(2):
                resp = await client.post(
                    chat_url,
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": _ANALYZE_SYSTEM},
                            {"role": "user", "content": user_msg},
                        ],
                        "max_tokens": 512,
                        "temperature": 0.3,
                    },
                )
                resp.raise_for_status()
                raw = resp.json()["choices"][0]["message"]["content"].strip()

                result = _extract_json(raw)
                if result is not None:
                    result.setdefault("verdict", "CAUTION")
                    result.setdefault("reasoning", "—")
                    result.setdefault("risks", [])
                    result.setdefault("confidence_delta", 0)
                    logger.info("berkaya/analyze: %s %s → %s", payload.pair, payload.direction, result["verdict"])
                    return result
                logger.warning("berkaya/analyze: JSON tidak valid (attempt %d): %s", attempt + 1, raw[:200])
        return {"error": f"LLM tidak return JSON valid: {raw[:200]}"}
    except Exception as exc:
        logger.error("berkaya/analyze error: %s", exc)
        return {"error": str(exc)}


_CANDLE_SYSTEM = """\
Kamu adalah technical analyst crypto berpengalaman.
Analisis data candlestick dan berikan pros/cons singkat.

SELALU respond dengan JSON valid:
{
  "trend": "BULLISH" | "BEARISH" | "RANGING",
  "confluence_score": <integer 0-100>,
  "patterns": ["nama pattern 1", ...],
  "pros": ["kelebihan setup 1", "kelebihan setup 2"],
  "cons": ["kelemahan 1", "kelemahan 2"],
  "ttl_seconds": 600
}

Aturan:
- pros/cons max 3 item masing-masing, singkat (1 kalimat)
- confluence_score: seberapa kuat sinyal 15m searah dengan 1H (0=contra, 100=sangat kuat)
- Jika 1H tidak ada, confluence_score hanya berdasarkan 15m saja (max 70)
"""


class CandleAnalyzePayload(BaseModel):
    pair: str
    direction: str = "LONG"
    candles_15m: list[dict]
    candles_1h:  list[dict] = []
    volume_ratio: float | None = None
    funding_rate: float | None = None
    oi_delta_pct: float | None = None
    session_name: str | None = None


def _describe_candles(candles: list[dict], label: str, n: int = 5) -> str:
    if not candles:
        return ""
    last = candles[-n:]
    lines = [f"\nCandle {label} (terbaru {len(last)} dari {len(candles)} total):"]
    for i, c in enumerate(last):
        o, h, l, cl = float(c["open"]), float(c["high"]), float(c["low"]), float(c["close"])
        body  = cl - o
        wick_up  = h - max(o, cl)
        wick_dn  = min(o, cl) - l
        color = "🟢" if body >= 0 else "🔴"
        lines.append(
            f"  [{i+1}] {color} O={o:.4f} C={cl:.4f} "
            f"body={body:+.4f} wick↑={wick_up:.4f} wick↓={wick_dn:.4f}"
        )
    return "\n".join(lines)


@_router.post("/candle/analyze")
async def analyze_candles(payload: CandleAnalyzePayload) -> dict:
    """
    Analisis pola candlestick 15m + confluence dengan 1H via LLM lokal.
    Return: {trend, confluence_score, patterns, pros, cons, ttl_seconds}
    """
    llm_info = _get_default_llm()
    if not llm_info:
        return {"error": "Tidak ada LLM endpoint — atur di Settings Odysseus."}

    llm_url, model = llm_info

    desc_15m = _describe_candles(payload.candles_15m, "15m", n=5)
    desc_1h  = _describe_candles(payload.candles_1h,  "1H",  n=3) if payload.candles_1h else ""

    htf_note = (
        f"Data 1H tersedia ({len(payload.candles_1h)} candle) — analisis confluence 15m vs 1H."
        if payload.candles_1h
        else "Data 1H tidak tersedia — hanya analisis 15m (confluence max 70)."
    )

    ctx_parts: list[str] = []
    if payload.volume_ratio is not None and payload.volume_ratio >= 2.0:
        ctx_parts.append(
            f"Volume spike: {payload.volume_ratio:.1f}× rata-rata "
            f"({'EXTREME' if payload.volume_ratio >= 4.0 else 'HIGH' if payload.volume_ratio >= 3.0 else 'MODERATE'}) "
            f"— kemungkinan momentum kuat atau short/long squeeze"
        )
    if payload.funding_rate is not None:
        fr_pct = payload.funding_rate * 100
        if fr_pct < -0.005:
            ctx_parts.append(f"Funding Rate: {fr_pct:+.4f}% (short dominan — ripe for short squeeze)")
        elif fr_pct > 0.005:
            ctx_parts.append(f"Funding Rate: {fr_pct:+.4f}% (long dominan — ripe for long squeeze)")
        else:
            ctx_parts.append(f"Funding Rate: {fr_pct:+.4f}% (netral)")
    if payload.oi_delta_pct is not None:
        ctx_parts.append(
            f"OI delta: {payload.oi_delta_pct:+.1f}% "
            f"({'likuidasi masif — posisi diclose paksa' if payload.oi_delta_pct < -2 else 'akumulasi — posisi bertambah' if payload.oi_delta_pct > 1 else 'stabil'})"
        )
    session_hint = {
        "US":     "Sedang jam US session (20:00–01:00 WIB) — volume tertinggi, breakout/spike lebih reliable.",
        "ASIA":   "Sedang jam Asia session (07:00–10:00 WIB) — volume moderate.",
        "EUROPE": "Sedang jam Europe session (14:00–20:00 WIB) — volume moderate.",
        "OFF":    "Di luar jam aktif utama (OFF session) — volume biasanya tipis, waspada false breakout.",
    }.get(payload.session_name or "")
    if session_hint:
        ctx_parts.append(session_hint)
    market_context = ("\n\nKonteks pasar real-time:\n" + "\n".join(f"- {p}" for p in ctx_parts)) if ctx_parts else ""

    user_msg = f"""Analisis setup ini:

Pair: {payload.pair} | Direction: {payload.direction}
{htf_note}
{desc_15m}
{desc_1h}{market_context}

Berikan verdict trend, pros, cons, patterns yang terdeteksi, dan confluence score."""

    chat_url = llm_url.rstrip("/")
    if not chat_url.endswith("/chat/completions"):
        if chat_url.endswith("/v1"):
            chat_url = chat_url[:-3]
        chat_url = chat_url.rstrip("/") + "/v1/chat/completions"

    raw = ""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            for attempt in range(2):
                resp = await client.post(
                    chat_url,
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": _CANDLE_SYSTEM},
                            {"role": "user",   "content": user_msg},
                        ],
                        "max_tokens": 400,
                        "temperature": 0.2,
                    },
                )
                resp.raise_for_status()
                raw = resp.json()["choices"][0]["message"]["content"].strip()

                result = _extract_json(raw)
                if result is not None:
                    result.setdefault("trend", "RANGING")
                    result.setdefault("confluence_score", 50)
                    result.setdefault("patterns", [])
                    result.setdefault("pros", [])
                    result.setdefault("cons", [])
                    result.setdefault("ttl_seconds", 600)
                    logger.info(
                        "berkaya/candle/analyze: %s %s → %s conf=%d",
                        payload.pair, payload.direction, result["trend"], result["confluence_score"],
                    )
                    return result
                logger.warning("berkaya/candle/analyze: JSON tidak valid (attempt %d): %s", attempt + 1, raw[:200])
        return {"error": f"LLM tidak return JSON valid: {raw[:200]}"}
    except Exception as exc:
        logger.error("berkaya/candle/analyze error: %s", exc)
        return {"error": str(exc)}


class BrainSavePayload(BaseModel):
    text: str
    category: str = "fact"
    source: str = "berkaya_daily_brief"


@_router.post("/brain/save")
async def save_to_brain(payload: BrainSavePayload) -> dict:
    """Simpan teks ke Odysseus memory (brain). Tersedia untuk semua LLM call berikutnya."""
    if _memory_manager is None:
        return {"ok": False, "error": "Memory manager tidak tersedia di Odysseus."}

    text = payload.text.strip()
    if not text:
        return {"ok": False, "error": "Teks kosong."}

    # Cek duplikat exact
    all_mem = _memory_manager.load_all()
    dupes = _memory_manager.find_duplicates(text, all_mem)
    if dupes:
        return {"ok": False, "duplicate": True, "message": "Memory identik sudah ada."}

    entry = _memory_manager.add_entry(text, source=payload.source, category=payload.category)
    all_mem.append(entry)
    _memory_manager.save(all_mem)

    logger.info("berkaya/brain: memory disimpan [%s] %.60s…", payload.category, text)
    return {"ok": True, "id": entry["id"], "category": entry["category"]}


class BrainSearchPayload(BaseModel):
    query: str
    limit: int = 3


@_router.post("/brain/search")
async def search_brain(payload: BrainSearchPayload) -> dict:
    """Cari memory yang relevan dengan query — historis setup serupa untuk konteks LLM."""
    if _memory_manager is None:
        return {"memories": [], "total": 0}

    all_mem = _memory_manager.load_all()
    relevant = _memory_manager.get_relevant_memories(
        payload.query, all_mem, threshold=0.05, max_items=payload.limit
    )
    return {"memories": relevant, "total": len(relevant)}


@_router.get("/daily_brief")
async def get_daily_brief() -> dict:
    """Return cached daily brief. {cached: false} jika belum ada atau expired."""
    if _brief_cache and time.time() < _brief_expires:
        age = int(time.time() - (_brief_expires - _BRIEF_TTL))
        ttl_left = int(_brief_expires - time.time())
        return {**_brief_cache, "cached": True, "age_seconds": age, "ttl_seconds": ttl_left}
    return {"cached": False}


async def _do_brief_refresh() -> dict:
    """Core logic refresh brief — dipanggil oleh endpoint dan scheduler."""
    global _brief_cache, _brief_expires

    headlines = await _fetch_headlines()
    if not headlines:
        logger.warning("berkaya/brief: tidak ada headlines dari RSS feed")
        return {"error": "Gagal fetch berita dari semua RSS feed."}

    llm_info = _get_default_llm()
    if llm_info:
        llm_url, model = llm_info
        result = await _brief_analyze(headlines, llm_url, model)
    else:
        result = {"sentiment": "NEUTRAL", "summary": "LLM tidak tersedia — headlines saja.", "veto": False}

    now = time.time()
    _brief_cache = {
        **result,
        "headlines": headlines,
        "headline_count": len(headlines),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    _brief_expires = now + _BRIEF_TTL
    _ctx_cache.clear()

    logger.info("berkaya/brief: refresh selesai → %s (%d headlines)", result["sentiment"], len(headlines))
    return {**_brief_cache, "cached": True, "age_seconds": 0, "ttl_seconds": int(_BRIEF_TTL)}


@_router.post("/daily_brief/refresh")
async def refresh_daily_brief() -> dict:
    """Fetch berita kripto terbaru dari RSS + analisis LLM. Cache 15 menit."""
    return await _do_brief_refresh()


def _start_scheduler() -> None:
    """Jalankan background thread yang auto-refresh brief tiap 15 menit."""
    global _scheduler_thread
    if _scheduler_thread and _scheduler_thread.is_alive():
        return

    async def _loop() -> None:
        # Delay singkat agar Odysseus fully ready dulu
        await asyncio.sleep(10)
        # Fetch awal saat startup jika cache masih kosong
        if _brief_cache is None:
            logger.info("berkaya/brief: initial fetch saat startup…")
            await _do_brief_refresh()
        # Loop auto-refresh setiap _BRIEF_TTL detik
        while True:
            await asyncio.sleep(_BRIEF_TTL)
            try:
                await _do_brief_refresh()
            except Exception as exc:
                logger.warning("berkaya/brief: auto-refresh gagal: %s", exc)

    def _run() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_loop())

    _scheduler_thread = threading.Thread(target=_run, daemon=True, name="brief-scheduler")
    _scheduler_thread.start()
    logger.info("berkaya/brief: scheduler dimulai (interval=%.0fs)", _BRIEF_TTL)


_POSITION_HEALTH_SYSTEM = """Kamu adalah risk advisor posisi trading crypto untuk sistem Berkaya.
Berikan assessment jujur tanpa bias — apakah posisi ini layak HOLD, perlu CAUTIOUS (waspada), atau harus EXIT segera.

Panduan verdict:
- HOLD    = kondisi masih oke, posisi tetap open, tidak ada sinyal bahaya kuat.
- CAUTIOUS = ada tekanan negatif (F&G rendah, HTF berlawanan, RSI ekstrem, momentum melemah) — perlu pantau ketat, belum harus exit.
- EXIT    = risiko nyata dan nyata posisi jadi loss: regime bear kuat + HTF berlawanan + momentum negatif, atau pct_to_sl < 0.5%. Harus ada dasar kuat — jangan exit hanya karena streak CAUTIOUS saja.

Kalibrasi mandiri (WAJIB digunakan):
- Lihat bagian "AKURASI HISTORIS C4" — itulah rekam jejak prediksi kamu sebelumnya.
- Semakin rendah akurasi EXIT historismu, semakin ketat syarat yang kamu butuhkan sebelum output EXIT.
- Confidence yang kamu tulis harus mencerminkan keyakinan nyata: jika akurasi historismu rendah atau datanya sedikit, tulis confidence moderat (0.50–0.70), bukan tinggi.
- Streak CAUTIOUS panjang bisa menjadi faktor pendukung EXIT — tapi hanya jika kondisi teknikal+macro memang terus memburuk. Jika ada tanda recovery, tetap CAUTIOUS atau turunkan ke HOLD.

Selalu pertimbangkan pct_to_sl — jika < 0.5%, prioritaskan proteksi modal.

WAJIB: Tulis reasoning dalam Bahasa Indonesia murni. Dilarang mencampur bahasa lain.

Jawab HANYA dalam format JSON satu baris tanpa markdown:
{"verdict": "HOLD|CAUTIOUS|EXIT", "confidence": 0.0-1.0, "reasoning": "1-2 kalimat singkat dalam Bahasa Indonesia murni"}"""


class PositionHealthPayload(BaseModel):
    pair: str
    direction: str
    entry_price: float
    mark_price: float
    pnl_pct: float
    pct_to_tp: float
    pct_to_sl: float
    rsi_15m: float = 50.0
    htf_trend: str = "—"
    htf_trend_4h: str = "—"
    htf_trend_1d: str = "—"
    htf_trend_1w: str = "—"
    session: str = "OFF"
    regime: str = "UNKNOWN"
    fear_greed: float = 50.0
    brain_context: str = "—"
    order_blocks: str = ""
    verdict_history: list[str] = []
    verdict_accuracy: list[dict] = []


@_router.post("/position/health")
async def analyze_position_health(payload: PositionHealthPayload) -> dict:
    """
    Verdict HOLD/CAUTIOUS/EXIT untuk posisi aktif berdasarkan konteks lengkap.
    Dipanggil oleh watcher/loop.py tiap trigger (price_delta, time_fallback, dsb).
    """
    llm_info = _get_default_llm()
    if not llm_info:
        return {"error": "Tidak ada LLM endpoint — atur di Settings Odysseus."}

    llm_url, model = llm_info

    # Streak: berapa kali verdict terbaru berulang berturut-turut (newest-first list)
    streak_type  = payload.verdict_history[0] if payload.verdict_history else ""
    streak_count = 0
    for v in payload.verdict_history:
        if v == streak_type:
            streak_count += 1
        else:
            break
    streak_str = f"{streak_count}× {streak_type} berturut" if streak_count > 1 else (streak_type or "—")

    history_str = " → ".join(payload.verdict_history) if payload.verdict_history else "Belum ada riwayat"

    # Akurasi historis — hanya data yang sudah ter-link ke outcome order
    if payload.verdict_accuracy:
        acc_lines = []
        for row in payload.verdict_accuracy:
            total = row.get("total", 0)
            wins  = row.get("wins", 0)
            loss  = row.get("losses", total - wins)
            acc_pct = f"{wins / total * 100:.0f}%" if total > 0 else "?"
            acc_lines.append(f"- {row['verdict']:8s}: {wins} benar / {loss} salah dari {total} data → {acc_pct}")
        accuracy_str = "\n".join(acc_lines)
    else:
        accuracy_str = "Belum ada data historis (posisi pertama atau verdict belum dievaluasi)."

    user_msg = f"""=== POSISI AKTIF ===
{payload.pair} {payload.direction} | Entry: ${payload.entry_price:.4f} | Mark: ${payload.mark_price:.4f} | PnL: {payload.pnl_pct:+.2f}%
Jarak ke TP: {payload.pct_to_tp:.2f}% | Jarak ke SL: {payload.pct_to_sl:.2f}%

=== TEKNIKAL ===
RSI 15m: {payload.rsi_15m:.1f} | Session: {payload.session}
HTF Trend: 1H={payload.htf_trend} | 4H={payload.htf_trend_4h} | 1D={payload.htf_trend_1d} | 1W={payload.htf_trend_1w}
Order Blocks (SMC): {payload.order_blocks or "tidak ada data"}

=== MACRO ===
Regime: {payload.regime} | Fear & Greed: {payload.fear_greed:.0f}/100

=== BRAIN RECALL (referensi saja — tetap tulis reasoning dalam Bahasa Indonesia) ===
{payload.brain_context}

=== AKURASI HISTORIS C4 (gunakan untuk kalibrasi confidence kamu) ===
{accuracy_str}

=== STREAK & RIWAYAT VERDICT ===
Streak saat ini: {streak_str}
Urutan (terbaru → terlama): {history_str}

Berikan verdict untuk posisi ini. Tulis reasoning dalam Bahasa Indonesia murni."""

    chat_url = llm_url.rstrip("/")
    if not chat_url.endswith("/chat/completions"):
        if chat_url.endswith("/v1"):
            chat_url = chat_url[:-3]
        chat_url = chat_url.rstrip("/") + "/v1/chat/completions"

    raw = ""
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            for attempt in range(2):
                resp = await client.post(
                    chat_url,
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": _POSITION_HEALTH_SYSTEM},
                            {"role": "user",   "content": user_msg},
                        ],
                        "max_tokens": 200,
                        "temperature": 0.2,
                    },
                )
                resp.raise_for_status()
                raw = resp.json()["choices"][0]["message"]["content"].strip()

                result = _extract_json(raw)
                if result is not None:
                    verdict = result.get("verdict", "HOLD").upper()
                    if verdict not in ("HOLD", "CAUTIOUS", "EXIT"):
                        verdict = "HOLD"
                    result["verdict"] = verdict
                    result.setdefault("confidence", 0.5)
                    result.setdefault("reasoning", "—")
                    try:
                        result["confidence"] = float(result["confidence"])
                        # Cap at 0.90 — 1.0 (100%) is never meaningful for an uncalibrated LLM.
                        # Escalation in watcher/loop.py can add +0.15, so max after escalation = 0.90.
                        result["confidence"] = max(0.0, min(0.90, result["confidence"]))
                    except Exception:
                        result["confidence"] = 0.5
                    logger.info(
                        "berkaya/position/health: %s %s → %s conf=%.0f%%",
                        payload.pair, payload.direction, result["verdict"], result["confidence"] * 100,
                    )
                    return result
                logger.warning(
                    "berkaya/position/health: JSON tidak valid (attempt %d): %s",
                    attempt + 1, raw[:200],
                )
        return {"error": f"LLM tidak return JSON valid: {raw[:200]}"}
    except Exception as exc:
        logger.error("berkaya/position/health error: %s", exc)
        return {"error": str(exc)}


def setup_berkaya_routes(memory_manager=None) -> APIRouter:
    global _memory_manager
    _memory_manager = memory_manager
    _start_scheduler()
    return _router
