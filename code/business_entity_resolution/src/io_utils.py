import glob
import os

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv

COLS = ["entity_id", "business_name", "business_address", "country"]


def _read_tsv(path):
    with open(path, "rb") as fh:
        header = fh.readline().decode("utf-8-sig").rstrip("\r\n").split("\t")
    header = [c.strip() for c in header]
    t = pacsv.read_csv(
        path,
        read_options=pacsv.ReadOptions(use_threads=True, block_size=1 << 26,
                                       column_names=header, skip_rows=1),
        parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False, newlines_in_values=False),
        convert_options=pacsv.ConvertOptions(column_types={c: pa.string() for c in header},
                                             strings_can_be_null=False))
    return t.to_pandas().fillna("")


def read_source(path):
    df = _read_tsv(path)
    for c in COLS:
        if c not in df.columns:
            df[c] = ""
    df = df[COLS].copy()
    for c in COLS:
        df[c] = df[c].astype(str).str.strip()
    return df


def read_ground_truth(path):
    df = _read_tsv(path)
    if "matched_entity_ids" not in df.columns:
        df["matched_entity_ids"] = ""
    return df[["source1_entity_id", "matched_entity_ids"]]


def _find_file(root, name):
    hits = sorted(glob.glob(os.path.join(root, "**", name), recursive=True), key=len)
    return hits[0] if hits else None


def find_data_dir(start_dirs):
    for d in start_dirs:
        if d and os.path.isdir(d):
            hit = _find_file(d, "train_source1.tsv")
            if hit:
                return os.path.dirname(os.path.dirname(hit)) \
                    if os.path.basename(os.path.dirname(hit)) == "train" else os.path.dirname(hit)
    return None


def load_split(data_dir, split):
    paths = [_find_file(data_dir, f"{split}_source{i}.tsv") for i in (1, 2, 3)]
    missing = [f"{split}_source{i}.tsv" for i, p in zip((1, 2, 3), paths) if p is None]
    if missing:
        raise FileNotFoundError(f"{missing} not found under {data_dir}")
    s = [read_source(p) for p in paths]
    gt_path = _find_file(data_dir, f"{split}_ground_truth.tsv")
    gt = read_ground_truth(gt_path) if gt_path else None
    print(f"[data] {split}: " + ", ".join(os.path.relpath(p, data_dir) for p in paths)
          + (f", {os.path.relpath(gt_path, data_dir)}" if gt_path else ""))
    return s[0], s[1], s[2], gt


def profile(split, s1, s2, s3, gt):
    print(f"[profile] {split}")
    for name, df in (("source1", s1), ("source2", s2), ("source3", s3)):
        empty = {c: f"{(df[c] == '').mean():.1%}" for c in COLS[1:]}
        countries = df.country.replace("", "<empty>").value_counts().to_dict()
        print(f"  {name}: {len(df)} rows, countries={countries}, empty fields={empty}")
        dup = df.entity_id.duplicated().sum()
        if dup:
            print(f"  WARNING {name}: {dup} duplicate entity_ids")
    if gt is not None:
        n = gt.matched_entity_ids.map(lambda v: len([x for x in v.split(",") if x.strip()]))
        print(f"  ground truth: {len(gt)} rows, singletons={(n == 0).mean():.1%}, "
              f"matches/entity mean={n.mean():.2f} max={n.max()}")
        unk = set(s1.entity_id) - set(gt.source1_entity_id)
        if unk:
            print(f"  note: {len(unk)} Source 1 entities have no ground-truth row -> treated as singletons")


def write_lists(path, s1_ids, lists, col):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(f"source1_entity_id\t{col}\n")
        for e in s1_ids:
            ids = list(dict.fromkeys(lists.get(e, [])))
            fh.write(f"{e}\t{','.join(ids)}\n")


def validate(matching_path, candidate_path, s1, s2, s3):
    issues = []
    s1_ids = set(s1.entity_id)
    valid_t = set(s2.entity_id) | set(s3.entity_id)
    parsed = {}
    for path, col in [(matching_path, "matched_entity_ids"),
                      (candidate_path, "candidate_entity_ids")]:
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        if list(df.columns) != ["source1_entity_id", col]:
            issues.append(f"{path}: bad header {list(df.columns)}")
            continue
        if df.source1_entity_id.duplicated().any():
            issues.append(f"{path}: duplicate source1_entity_id rows")
        missing = s1_ids - set(df.source1_entity_id)
        extra = set(df.source1_entity_id) - s1_ids
        if missing:
            issues.append(f"{path}: {len(missing)} Source 1 entities missing")
        if extra:
            issues.append(f"{path}: {len(extra)} unknown Source 1 ids")
        lists = {}
        for e, v in zip(df.source1_entity_id, df[col]):
            ids = [x for x in v.split(",") if x] if v else []
            if len(ids) != len(set(ids)):
                issues.append(f"{path}: duplicate ids in list for {e}")
            bad = [x for x in ids if x not in valid_t or not x.startswith(("S2-", "S3-"))]
            if bad:
                issues.append(f"{path}: invalid ids for {e}: {bad[:3]}")
            lists[e] = set(ids)
        parsed[col] = lists
    if len(parsed) == 2:
        m, c = parsed["matched_entity_ids"], parsed["candidate_entity_ids"]
        leak = sum(len(v - c.get(k, set())) for k, v in m.items())
        if leak:
            issues.append(f"{leak} matched ids are not in candidate_pairs (should be a subset)")
    return issues
