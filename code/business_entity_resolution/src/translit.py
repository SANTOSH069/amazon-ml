import re

SCRIPT_BASES = [0x0900, 0x0980, 0x0A00, 0x0A80, 0x0B00, 0x0B80, 0x0C00, 0x0C80, 0x0D00]
INDIC_RE = re.compile(r"[ऀ-෿]")

_VOWELS = {0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u", 0x0B: "ri",
           0x0C: "li", 0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o",
           0x13: "o", 0x14: "au", 0x60: "ri", 0x61: "li"}
_MATRAS = {0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u", 0x43: "ri", 0x44: "ri",
           0x45: "e", 0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o",
           0x4C: "au", 0x62: "li", 0x63: "li", 0x56: "ai", 0x57: "au"}
_CONS = {0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "n", 0x1A: "ch", 0x1B: "chh",
         0x1C: "j", 0x1D: "jh", 0x1E: "n", 0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh",
         0x23: "n", 0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n",
         0x2A: "p", 0x2B: "ph", 0x2C: "b", 0x2D: "bh", 0x2E: "m", 0x2F: "y", 0x30: "r",
         0x31: "r", 0x32: "l", 0x33: "l", 0x34: "l", 0x35: "v", 0x36: "sh", 0x37: "sh",
         0x38: "s", 0x39: "h", 0x58: "q", 0x59: "kh", 0x5A: "g", 0x5B: "z", 0x5C: "d",
         0x5D: "rh", 0x5E: "f", 0x5F: "y"}
_NASAL = {0x01: "n", 0x02: "n", 0x03: "h", 0x70: "n"}
_VIRAMA = 0x4D
_EXTRA = {0x0D7A: "n", 0x0D7B: "n", 0x0D7C: "r", 0x0D7D: "l", 0x0D7E: "l", 0x0D7F: "k",
          0x09CE: "t", 0x0D54: "m", 0x0D55: "y", 0x0D56: "l"}


def _offset(ch):
    cp = ord(ch)
    if cp in _EXTRA:
        return None, _EXTRA[cp]
    if 0x0900 <= cp <= 0x0DFF:
        base = SCRIPT_BASES[(cp - 0x0900) // 0x80]
        return cp - base, None
    return None, None


def romanize_word(word):
    out = []
    pending = False
    for ch in word:
        off, extra = _offset(ch)
        if extra is not None:
            if pending:
                out.append("a")
            out.append(extra)
            pending = False
            continue
        if off is None:
            if pending:
                out.append("a")
                pending = False
            out.append(ch)
            continue
        if off in _CONS:
            if pending:
                out.append("a")
            out.append(_CONS[off])
            pending = True
        elif off in _MATRAS:
            out.append(_MATRAS[off])
            pending = False
        elif off == _VIRAMA:
            pending = False
        elif off in _VOWELS:
            if pending:
                out.append("a")
            out.append(_VOWELS[off])
            pending = False
        elif off in _NASAL:
            if pending:
                out.append("a")
            out.append(_NASAL[off])
            pending = False
        elif 0x66 <= off <= 0x6F:
            if pending:
                out.append("a")
            out.append(str(off - 0x66))
            pending = False
    s = "".join(out)
    return s.replace("aa", "a").replace("ii", "i").replace("uu", "u").replace("ee", "e")


def has_indic(s):
    return bool(INDIC_RE.search(s))
