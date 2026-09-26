"""
normalize.py — Fast text normalization for business names and addresses.

Optimised for millions of rows: pre-compiled regex, minimal per-char iteration.
"""

import re
import unicodedata

# ── Legal suffix normalization (compiled) ────────────────────────────────────
_SUFFIX_PATTERNS = [
    re.compile(r'\bprivate\s+limited\b'),
    re.compile(r'\bpvt\.?\s*ltd\.?\b'),
    re.compile(r'\blimited\b'),
    re.compile(r'\bltd\.?\b'),
    re.compile(r'\bcorporation\b'),
    re.compile(r'\bcorp\.?\b'),
    re.compile(r'\bincorporated\b'),
    re.compile(r'\binc\.?\b'),
    re.compile(r'\bllc\.?\b'),
    re.compile(r'\bllp\.?\b'),
    re.compile(r'\bcompany\b'),
    re.compile(r'\bsarl\.?\b'),
    re.compile(r'\beurl\.?\b'),
    re.compile(r'\bpvt\.?\b'),
]

# ── Address abbreviation map ─────────────────────────────────────────────────
_ADDR_ABBR = {
    'rd': 'road', 'st': 'street', 'ave': 'avenue', 'blvd': 'boulevard',
    'dr': 'drive', 'ln': 'lane', 'ct': 'court', 'pl': 'place',
    'sq': 'square', 'pkwy': 'parkway', 'hwy': 'highway', 'cir': 'circle',
    'trl': 'trail', 'ter': 'terrace', 'apt': 'apartment', 'ste': 'suite',
    'bldg': 'building', 'fl': 'floor', 'rm': 'room',
    'n': 'north', 's': 'south', 'e': 'east', 'w': 'west',
    'ne': 'northeast', 'nw': 'northwest', 'se': 'southeast', 'sw': 'southwest',
    'mt': 'mount', 'ft': 'fort', 'pt': 'point',
}

# Pre-compiled regexes for speed
_RE_POSSESSIVE = re.compile(r"'s\b")
_RE_PUNCT = re.compile(r'[^\w\s-]')
_RE_WS = re.compile(r'\s+')
_RE_ADDR_UNIT = re.compile(
    r'\b(?:unit|apartment|apt|suite|ste|room|rm|floor|fl)\s*[#]?\s*\w+', re.I)


def _to_ascii(text: str) -> str:
    """Fast NFKD → strip combining marks → encode ASCII (drop rest)."""
    nfkd = unicodedata.normalize('NFKD', text)
    # This is the fastest way: encode to ascii ignoring errors
    # Combining marks (accents) are non-ASCII after NFKD, so they get dropped.
    return nfkd.encode('ascii', 'ignore').decode('ascii')


# Common domain and trade name prefixes/suffixes
_RE_DOMAIN = re.compile(r'\.(?:com|org|net|in|fr|co|biz|info)\b')
_RE_DBA = re.compile(r'\b(?:dba|formerly|aka)\b')
_RE_LEET_AT = re.compile(r'@')
_RE_LEADING_ZEROS = re.compile(r'\b0+(\d+)\b')

def normalize_text(text: str) -> str:
    """Basic text normalization: unicode→ascii, lowercase, collapse whitespace."""
    if not text or not isinstance(text, str):
        return ''
    text = _to_ascii(text)
    text = text.lower()
    text = text.replace('&', ' and ')
    text = _RE_LEET_AT.sub('a', text)
    text = _RE_POSSESSIVE.sub('', text)
    text = _RE_PUNCT.sub(' ', text)
    text = _RE_WS.sub(' ', text).strip()
    return text


def normalize_name(name: str) -> str:
    """Normalize a business name: strip domains, dba, legal suffixes."""
    text = normalize_text(name)
    text = _RE_DOMAIN.sub('', text)
    text = _RE_DBA.sub('', text)
    for pat in _SUFFIX_PATTERNS:
        text = pat.sub('', text)
    text = text.strip(' -')
    text = _RE_WS.sub(' ', text).strip()
    return text


def normalize_address(address: str) -> str:
    """Normalize an address: strip leading zeros on numbers, expand abbreviations."""
    text = normalize_text(address)
    # Strip leading zeros so 0337 -> 337, 0017560 -> 17560
    text = _RE_LEADING_ZEROS.sub(r'\1', text)
    tokens = text.split()
    tokens = [_ADDR_ABBR.get(t.rstrip('.,'), t) for t in tokens]
    text = ' '.join(tokens)
    text = _RE_ADDR_UNIT.sub('', text)
    text = _RE_WS.sub(' ', text).strip()
    return text


def name_tokens(name: str) -> set:
    """Get significant tokens from a normalized name."""
    _stop = {'the', 'of', 'and', 'a', 'an', 'in', 'at', 'for', 'to',
             'by', 'on', 'with', 'from', 'de', 'du', 'des', 'le', 'la', 'les'}
    return {t for t in name.split() if t not in _stop and len(t) > 1}


def address_tokens(addr: str) -> set:
    """Get significant tokens from a normalized address."""
    _stop = {'the', 'of', 'and', 'a', 'an', 'in', 'at', 'for', 'to',
             'by', 'on', 'near', 'behind', 'beside', 'opposite', 'next',
             'above', 'below', 'no', 'po', 'box'}
    return {t for t in addr.split() if t not in _stop and len(t) > 1}


def char_ngrams(text: str, n: int = 3) -> set:
    """Get character n-grams from text."""
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}
