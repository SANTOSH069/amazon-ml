import re
import unicodedata
from collections import Counter, defaultdict
from translit import has_indic, romanize_word

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
INDIC_RUN = re.compile(r"[\u0900-\u0DFF]+")


def fold(text):
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.encode("ascii", "ignore").decode("ascii").lower()


def to_latin(text, indic_words):
    if not has_indic(text):
        return text
    return INDIC_RUN.sub(
        lambda m: " " + (indic_words.get(m.group(0)) or romanize_word(m.group(0))) + " ", text)


def fix_leetspeak(word):
    if word.isdigit() or word.isalpha() or ORDINAL.match(word):
        if word.startswith("l") and len(word) > 2 and word[1] in "nmdt" and word not in ("lt", "ltd", "llc", "llp"):
            return "i" + word[1:]
        return word
    letters = sum(c.isalpha() for c in word)
    if letters >= 2 and len(word) - letters <= 2:
        return word.translate(LEET)
    return word


def clean_name(raw, indic_words):
    is_indic = has_indic(raw)
    text = fold(to_latin(raw, indic_words))
    is_website = bool(WEB_RE.search(text))
    if "|" in text:
        text = text.split("|")[0]
    is_alias = False
    alias = ALIAS_RE.search(text)
    if alias:
        real_name = text[alias.end():]
        if TOKEN.search(real_name):
            text, is_alias = real_name, True
    if is_website:
        text = re.sub(r"www\.|\.(?:com|co\.in|in|net|org|fr|biz)\b", " ", text)
    text = DOTTED.sub(r"\1", text)
    text = DOTTED.sub(r"\1", text)
    text = text.replace("&", " and ").replace("+", " and ")
    words, seen = [], set()
    for word in TOKEN.findall(text):
        word = fix_leetspeak(word)
        word = NAME_CANON.get(word, word)
        if word in HONORIFIC or word in seen:
            continue
        seen.add(word)
        words.append(word)
    core = [w for w in words if w not in LEGAL and w not in GENERIC] or words
    return " ".join(words), " ".join(core), is_alias, is_website, is_indic


NUMERO = re.compile(r"\b[Nn]\s*[°º]")


def clean_address(raw, indic_words, synonyms):
    text = fold(to_latin(NUMERO.sub(" no ", raw), indic_words)).replace("#", " no ")
    text = DOTTED.sub(r"\1", text)
    words, seen, numbers = [], set(), []
    for word in TOKEN.findall(text):
        word = ADDR_CANON.get(word, word)
        word = synonyms.get(word, word)
        if word in ADDR_STOP or word in seen:
            continue
        seen.add(word)
        words.append(word)
        for digits in DIGITS.findall(word):
            digits = digits.lstrip("0") or "0"
            if digits not in numbers:
                numbers.append(digits)
    return words, numbers


def raw_name_words(raw):
    return [w for w in re.split(r"[^\w\u0900-\u0DFF]+", raw) if w]


def learn_indic_words(source1_names, target_names, min_count=1):
    votes = defaultdict(Counter)
    for source1_name, target_name in zip(source1_names, target_names):
        if not has_indic(target_name):
            continue
        latin_words = [NAME_CANON.get(w, w) for w in TOKEN.findall(fold(source1_name))]
        target_words = raw_name_words(target_name)
        if len(latin_words) != len(target_words):
            continue
        for target_word, latin_word in zip(target_words, latin_words):
            runs = INDIC_RUN.findall(target_word)
            if runs:
                votes[runs[0] if len(runs) == 1 else target_word][latin_word] += 1
    dictionary = {}
    for word, counts in votes.items():
        latin_word, count = counts.most_common(1)[0]
        if count >= min_count and count / sum(counts.values()) >= 0.5:
            dictionary[word] = latin_word
    return dictionary


def learn_address_synonyms(source1_word_lists, target_word_lists, min_count=30, min_share=0.6,
                           max_source1_share=0.1):
    df_source1, df_target = Counter(), Counter()
    seen, replaced = Counter(), defaultdict(Counter)
    for source1_words, target_words in zip(source1_word_lists, target_word_lists):
        source1_set, target_set = set(source1_words), set(target_words)
        df_source1.update(source1_set)
        df_target.update(target_set)
        missing = source1_set - target_set
        for word in target_set - source1_set:
            seen[word] += 1
            replaced[word].update(missing)
    synonyms = {}
    for word, count in seen.items():
        if (count < min_count or word.isdigit() or not replaced[word]
                or df_source1[word] > max_source1_share * df_target[word]):
            continue
        replacement, hits = replaced[word].most_common(1)[0]
        if hits / count >= min_share and not replacement.isdigit():
            synonyms[word] = replacement
    return synonyms
