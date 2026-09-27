import glob
import os

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]


def read_tsv(path):
    with open(path, "rb") as handle:
        header = handle.readline().decode("utf-8-sig").rstrip("\r\n").split("\t")
    header = [name.strip() for name in header]
    table = pacsv.read_csv(
        path,
        read_options=pacsv.ReadOptions(use_threads=True, block_size=1 << 26,
                                       column_names=header, skip_rows=1),
        parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False, newlines_in_values=False),
        convert_options=pacsv.ConvertOptions(column_types={name: pa.string() for name in header},
                                             strings_can_be_null=False))
    return table.to_pandas().fillna("")


def read_source(path):
    frame = read_tsv(path)
    for column in SOURCE_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    frame = frame[SOURCE_COLUMNS].copy()
    for column in SOURCE_COLUMNS:
        frame[column] = frame[column].astype(str).str.strip()
    return frame


def read_ground_truth(path):
    frame = read_tsv(path)
    if "matched_entity_ids" not in frame.columns:
        frame["matched_entity_ids"] = ""
    return frame[["source1_entity_id", "matched_entity_ids"]]


def find_file(root, name):
    hits = sorted(glob.glob(os.path.join(root, "**", name), recursive=True), key=len)
    return hits[0] if hits else None


def find_data_dir(search_dirs):
    for folder in search_dirs:
        if folder and os.path.isdir(folder):
            hit = find_file(folder, "train_source1.tsv")
            if hit:
                parent = os.path.dirname(hit)
                return os.path.dirname(parent) if os.path.basename(parent) == "train" else parent
    return None


def load_split(data_dir, split):
    paths = [find_file(data_dir, f"{split}_source{i}.tsv") for i in (1, 2, 3)]
    missing = [f"{split}_source{i}.tsv" for i, path in zip((1, 2, 3), paths) if path is None]
    if missing:
        raise FileNotFoundError(f"{missing} not found under {data_dir}")
    s1, s2, s3 = (read_source(path) for path in paths)
    truth_path = find_file(data_dir, f"{split}_ground_truth.tsv")
    ground_truth = read_ground_truth(truth_path) if truth_path else None
    found_paths = [path for path in paths if path is not None]
    if truth_path is not None:
        found_paths.append(truth_path)
    found = [os.path.relpath(path, data_dir) for path in found_paths]
    print(f"[data] {split}: " + ", ".join(found))
    return s1, s2, s3, ground_truth


def profile(split, s1, s2, s3, ground_truth):
    print(f"[profile] {split}")
    for name, frame in (("source1", s1), ("source2", s2), ("source3", s3)):
        empty = {column: f"{(frame[column] == '').mean():.1%}" for column in SOURCE_COLUMNS[1:]}
        countries = frame.country.replace("", "<empty>").value_counts().to_dict()
        print(f"  {name}: {len(frame)} rows, countries={countries}, empty fields={empty}")
        duplicates = frame.entity_id.duplicated().sum()
        if duplicates:
            print(f"  WARNING {name}: {duplicates} duplicate entity_ids")
    if ground_truth is not None:
        match_counts = ground_truth.matched_entity_ids.map(
            lambda ids: len([x for x in ids.split(",") if x.strip()]))
        print(f"  ground truth: {len(ground_truth)} rows, singletons={(match_counts == 0).mean():.1%}, "
              f"matches/entity mean={match_counts.mean():.2f} max={match_counts.max()}")
        without_row = set(s1.entity_id) - set(ground_truth.source1_entity_id)
        if without_row:
            print(f"  note: {len(without_row)} Source 1 entities have no ground-truth row "
                  f"and are treated as singletons")


def write_lists(path, source1_ids, lists, column):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(f"source1_entity_id\t{column}\n")
        for entity in source1_ids:
            ids = list(dict.fromkeys(lists.get(entity, [])))
            handle.write(f"{entity}\t{','.join(ids)}\n")


def validate(matching_path, candidate_path, s1, s2, s3):
    issues = []
    source1_ids = set(s1.entity_id)
    target_ids = set(s2.entity_id) | set(s3.entity_id)
    parsed = {}
    for path, column in [(matching_path, "matched_entity_ids"),
                         (candidate_path, "candidate_entity_ids")]:
        frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        if list(frame.columns) != ["source1_entity_id", column]:
            issues.append(f"{path}: bad header {list(frame.columns)}")
            continue
        if frame.source1_entity_id.duplicated().any():
            issues.append(f"{path}: duplicate source1_entity_id rows")
        missing = source1_ids - set(frame.source1_entity_id)
        unknown = set(frame.source1_entity_id) - source1_ids
        if missing:
            issues.append(f"{path}: {len(missing)} Source 1 entities missing")
        if unknown:
            issues.append(f"{path}: {len(unknown)} unknown Source 1 ids")
        lists = {}
        for entity, value in zip(frame.source1_entity_id, frame[column]):
            ids = [x for x in value.split(",") if x] if value else []
            if len(ids) != len(set(ids)):
                issues.append(f"{path}: duplicate ids in list for {entity}")
            invalid = [x for x in ids if x not in target_ids or not x.startswith(("S2-", "S3-"))]
            if invalid:
                issues.append(f"{path}: invalid ids for {entity}: {invalid[:3]}")
            lists[entity] = set(ids)
        parsed[column] = lists
    if len(parsed) == 2:
        matches, candidates = parsed["matched_entity_ids"], parsed["candidate_entity_ids"]
        outside = sum(len(ids - candidates.get(entity, set())) for entity, ids in matches.items())
        if outside:
            issues.append(f"{outside} matched ids are not in candidate_pairs (should be a subset)")
    return issues
