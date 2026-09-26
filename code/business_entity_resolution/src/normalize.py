import re
import unicodedata
from collections import Counter, defaultdict

from translit import INDIC_RE, has_indic, romanize_word

NAME_CANON = {
    "corp": "corporation", "corpn": "corporation", "co": "company", "cos": "company",
    "cie": "company", "compagnie": "company", "inc": "incorporated", "incorp": "incorporated",
    "lnc": "incorporated", "ltd": "limited", "ltda": "limited", "pvt": "private",
    "pte": "private", "prv": "private", "intl": "international", "mfg": "manufacturing",
    "svc": "services", "svcs": "services", "service": "services", "bros": "brothers",
    "assoc": "associates", "grp": "group", "inds": "industries", "mgmt": "management",
    "natl": "national", "ent": "enterprises", "tech": "technologies",
    "technology": "technologies", "sys": "systems", "univ": "university",
    "centre": "center", "ctr": "center", "et": "and", "n": "and", "st": "saint",
    "ste": "sainte", "mt": "mount", "societe": "societe", "sté": "societe",
    "etablissements": "etablissement", "ets": "etablissement",
}
LEGAL = {
    "incorporated", "corporation", "company", "limited", "private", "llc", "llp", "lp",
    "plc", "pllc", "pc", "pa", "opc", "sa", "sas", "sasu", "sarl", "eurl", "sci", "snc",
    "scop", "sca", "selarl", "gie", "gmbh", "ag", "bv", "nv", "pty", "societe",
}
GENERIC = {"services", "center", "partners", "labs", "one", "the", "com", "www", "india",
           "lndia", "usa", "france", "and", "of", "de", "du", "des", "la", "le", "les", "l", "d"}
HONORIFIC = {"mr", "mrs", "ms", "smt", "shri", "sri", "shree", "dr", "m", "s"}

ADDR_CANON = {
    "road": "rd", "street": "st", "str": "st", "avenue": "ave", "av": "ave", "avn": "ave",
    "boulevard": "blvd", "bd": "blvd", "boul": "blvd", "drive": "dr", "lane": "ln",
    "court": "ct", "circle": "cir", "place": "pl", "square": "sq", "sqr": "sq",
    "terrace": "ter", "trail": "trl", "highway": "hwy", "parkway": "pkwy", "way": "wy",
    "north": "n", "south": "s", "east": "e", "west": "w", "northeast": "ne",
    "northwest": "nw", "southeast": "se", "southwest": "sw", "saint": "st", "sainte": "ste",
    "mount": "mt", "fort": "ft", "building": "bldg", "floor": "fl", "flr": "fl",
    "first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th", "fifth": "5th",
    "nagar": "ngr", "nagr": "ngr", "marg": "mg", "sector": "sec", "sect": "sec",
    "colony": "col", "block": "blk", "blck": "blk", "cross": "crs", "main": "mn",
    "stage": "stg", "phase": "ph", "extension": "extn", "ext": "extn", "layout": "lyt",
    "chowk": "chk", "bazaar": "bzr", "bazar": "bzr", "market": "mkt", "station": "stn",
    "complex": "cmplx", "apartments": "apts", "tower": "twr", "towers": "twr",
    "opposite": "opp", "near": "nr", "behind": "bhd",
    "r": "rue", "allee": "all", "impasse": "imp", "chemin": "ch", "route": "rte",
    "cours": "crs", "faubourg": "fg", "quai": "qu", "residence": "res", "res": "res",
}
ADDR_STOP = {"no", "door", "h", "hno", "hn", "unit", "apt", "apartment", "suite", "ste",
             "nr", "opp", "bhd", "cdp", "city", "township", "null", "na", "pmb", "po", "box",
             "the", "of", "and", "at", "de", "du", "des", "la", "le", "les", "l", "d",
             "bis", "ter", "flat", "plot", "fl", "cedex"}

ALIAS_RE = re.compile(
    r"\b(?:d\s*/\s*b\s*/\s*a|d\.b\.a\.?|dba|doing business as|a\s*/\s*k\s*/\s*a|a\.k\.a\.?|aka|"
    r"also known as|f\s*/\s*k\s*/\s*a|f\.k\.a\.?|fka|formerly known as|formerly|nee|"
    r"trading as|t\s*/\s*a|operating as)\b")
WEB_RE = re.compile(r"www\.|\.com\b|\.co\.in\b|\.in\b|\.net\b|\.org\b|\.fr\b|\.biz\b|^\s*[#@]\w")
DOTTED = re.compile(r"\b([a-z0-9])\.(?=[a-z0-9]\b)")
TOKEN = re.compile(r"[a-z0-9]+")
LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b"})
ORDINAL = re.compile(r"^\d+(st|nd|rd|th)$")
DIGITS = re.compile(r"\d+")
INDIC_RUN = re.compile(r"[ऀ-෿]+")


def fold(s):
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.encode("ascii", "ignore").decode("ascii").lower()


def latinize(s, native):
    if not has_indic(s):
        return s
    return INDIC_RUN.sub(lambda m: " " + (native.get(m.group(0)) or romanize_word(m.group(0))) + " ", s)


def _fix_leet(t):
    if t.isdigit() or t.isalpha() or ORDINAL.match(t):
        if t.startswith("l") and len(t) > 2 and t[1] in "nmdt" and t not in ("lt", "ltd", "llc", "llp"):
            return "i" + t[1:]
        return t
    letters = sum(c.isalpha() for c in t)
    if letters >= 2 and len(t) - letters <= 2:
        return t.translate(LEET)
    return t


def name_parts(raw, native):
    script = has_indic(raw)
    s = fold(latinize(raw, native))
    web = bool(WEB_RE.search(s))
    if "|" in s:
        s = s.split("|")[0]
    alias = False
    m = ALIAS_RE.search(s)
    if m:
        after = s[m.end():]
        if TOKEN.search(after):
            s, alias = after, True
    if web:
        s = re.sub(r"www\.|\.(?:com|co\.in|in|net|org|fr|biz)\b", " ", s)
    s = DOTTED.sub(r"\1", s)
    s = DOTTED.sub(r"\1", s)
    s = s.replace("&", " and ").replace("+", " and ")
    toks, seen = [], set()
    for t in TOKEN.findall(s):
        t = NAME_CANON.get(_fix_leet(t), _fix_leet(t))
        if t in HONORIFIC or t in seen:
            continue
        seen.add(t)
        toks.append(t)
    core = [t for t in toks if t not in LEGAL and t not in GENERIC]
    if not core:
        core = toks
    return " ".join(toks), " ".join(core), alias, web, script


def addr_parts(raw, native, syn):
    s = fold(latinize(raw, native)).replace("n°", " no ").replace("#", " no ")
    s = DOTTED.sub(r"\1", s)
    toks, seen = [], set()
    nums = []
    for t in TOKEN.findall(s):
        t = ADDR_CANON.get(t, t)
        t = syn.get(t, t)
        if t in ADDR_STOP or t in seen:
            continue
        seen.add(t)
        toks.append(t)
        for d in DIGITS.findall(t):
            d = d.lstrip("0") or "0"
            if d not in nums:
                nums.append(d)
    return toks, nums


def _raw_name_tokens(raw):
    return [t for t in re.split(r"[^\wऀ-෿]+", raw) if t]


def learn_native_dict(s1_names, t_names, min_count=1):
    votes = defaultdict(Counter)
    for a, b in zip(s1_names, t_names):
        if not has_indic(b):
            continue
        ta = [NAME_CANON.get(t, t) for t in TOKEN.findall(fold(a))]
        tb = _raw_name_tokens(b)
        if len(ta) != len(tb):
            continue
        for x, y in zip(tb, ta):
            if INDIC_RE.search(x):
                votes[INDIC_RUN.findall(x)[0] if len(INDIC_RUN.findall(x)) == 1 else x][y] += 1
    out = {}
    for w, c in votes.items():
        y, n = c.most_common(1)[0]
        if n >= min_count and n / sum(c.values()) >= 0.5:
            out[w] = y
    return out


def learn_addr_synonyms(s1_tok_lists, t_tok_lists, min_count=30, min_p=0.6, max_rel_df=0.1):
    df1, dft = Counter(), Counter()
    cnt, co = Counter(), defaultdict(Counter)
    for a, b in zip(s1_tok_lists, t_tok_lists):
        sa, sb = set(a), set(b)
        df1.update(sa)
        dft.update(sb)
        missing = sa - sb
        for w in sb - sa:
            cnt[w] += 1
            co[w].update(missing)
    syn = {}
    for w, n in cnt.items():
        if n < min_count or w.isdigit() or df1[w] > max_rel_df * dft[w] or not co[w]:
            continue
        v, k = co[w].most_common(1)[0]
        if k / n >= min_p and not v.isdigit():
            syn[w] = v
    return syn
