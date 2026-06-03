import html
import json
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path


_STATUS_RE = re.compile(r"(?:twitter|x)\.com/[^/]+/status/(\d+)")
_LEADING_MENTIONS = re.compile(r"^(?:@\w+\s+)+")
_MONTHS = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], start=1)}


def load_js_array(path: Path) -> list:
    """Read a `window.YTD.<name>.partN = [ ... ]` file -> parsed JSON list. Tolerates BOM."""
    raw = path.read_text(encoding="utf-8-sig")
    return json.loads(raw[raw.index("["):])


def normalize_tweet(tweet: dict) -> tuple[str, str | None, int]:
    """Return (normalized_text, quote_of_id|None, media_count). Order matters."""
    text = tweet["full_text"]
    entities = tweet.get("entities") or {}
    ext = tweet.get("extended_entities") or {}
    quote_of_id = None

    for u in entities.get("urls") or []:
        link = u.get("url")
        if not link:
            continue
        # TODO: validate vs real archive
        m = _STATUS_RE.search(u.get("expanded_url", ""))
        if m:
            quote_of_id = quote_of_id or m.group(1)
            text = text.replace(link, "")
        else:
            text = text.replace(link, u.get("expanded_url") or link)

    media = ext.get("media") if ext.get("media") is not None else entities.get("media") or []
    media_count = len(media)
    for m_ in media:
        if m_.get("url"):
            text = text.replace(m_["url"], "")

    if tweet.get("in_reply_to_status_id_str") is not None:
        text = _LEADING_MENTIONS.sub("", text)

    text = html.unescape(text)
    text = " ".join(text.split())
    return text, quote_of_id, media_count


def parse_twitter_dt(s: str | None) -> datetime | None:
    """'Wed May 15 10:00:00 +0000 2024' -> tz-aware UTC datetime, or None if unparseable."""
    if not s:
        return None
    try:
        _wday, mon, day, clock, tz, year = s.split()
        hh, mm, ss = (int(x) for x in clock.split(":"))
        sign = 1 if tz[0] == "+" else -1
        offset = timezone(sign * timedelta(hours=int(tz[1:3]), minutes=int(tz[3:5])))
        return datetime(int(year), _MONTHS[mon], int(day), hh, mm, ss,
                        tzinfo=offset).astimezone(timezone.utc)
    except (ValueError, KeyError, IndexError):
        return None
