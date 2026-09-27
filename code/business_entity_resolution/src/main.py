import argparse
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from io_utils import find_data_dir, load_split, profile, validate, write_lists
from resolver import Resolver


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None,
                    help="folder containing train/ and test/ (auto-detected if omitted)")
    parser.add_argument("--out-dir", default=None, help="default: output/ next to the dataset")
    parser.add_argument("--model", default=None, help="default: model/resolver.pkl in the project")
    parser.add_argument("--mode", choices=["all", "fit", "predict"], default="all")
    parser.add_argument("--top-k", type=int, default=5, help="blocking: Source 1 candidates kept per target")
    parser.add_argument("--target-recall", type=float, default=0.995,
                    help="share of blocking true pairs the candidate filter must keep")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    started = time.time()
    here = os.path.dirname(os.path.abspath(__file__))
    project = os.path.dirname(here)
    submission_root = os.path.dirname(os.path.dirname(project))
    data_dir = args.data_dir or find_data_dir([os.getcwd(), project, submission_root])
    if data_dir is None:
        print("No dataset found. Pass --data-dir <folder containing train/ and test/>.")
        return 2
    data_dir = os.path.abspath(data_dir)
    out_dir = args.out_dir or os.path.join(os.path.dirname(data_dir), "output")
    model_path = args.model or os.path.join(project, "model", "resolver.pkl")
    print(f"[data] using {data_dir}", flush=True)

    if args.mode in ("all", "fit"):
        s1, s2, s3, ground_truth = load_split(data_dir, "train")
        if ground_truth is None:
            print("train_ground_truth.tsv not found - it is required for training.")
            return 2
        profile("train", s1, s2, s3, ground_truth)
        resolver = Resolver(top_k=args.top_k, target_recall=args.target_recall, seed=args.seed)
        resolver.fit(s1, s2, s3, ground_truth)
        resolver.log = None
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as handle:
            pickle.dump(resolver, handle, protocol=5)
        print(f"[fit] saved {model_path}  ({time.time() - started:.0f}s)", flush=True)
        del s1, s2, s3, ground_truth

    if args.mode in ("all", "predict"):
        with open(model_path, "rb") as handle:
            resolver = pickle.load(handle)
        resolver.log = lambda *parts: print(*parts, flush=True)
        s1, s2, s3, _ = load_split(data_dir, "test")
        profile("test", s1, s2, s3, None)
        candidates, matches = resolver.predict(s1, s2, s3)
        matching_path = os.path.join(out_dir, "matching_results.tsv")
        candidate_path = os.path.join(out_dir, "candidate_pairs.tsv")
        ids = s1.entity_id.tolist()
        write_lists(matching_path, ids, matches, "matched_entity_ids")
        write_lists(candidate_path, ids, candidates, "candidate_entity_ids")
        issues = validate(matching_path, candidate_path, s1, s2, s3)
        n_matches = sum(len(v) for v in matches.values())
        n_candidates = sum(len(v) for v in candidates.values())
        print(f"[test] S1={len(ids)} candidates={n_candidates} ({n_candidates / max(len(ids), 1):.2f}/S1) "
              f"matches={n_matches} ({n_matches / max(len(ids), 1):.2f}/S1) "
              f"non-empty={sum(bool(v) for v in matches.values())}")
        print("[validate] PASS" if not issues else "[validate] FAIL\n  " + "\n  ".join(issues))
        print(f"done in {time.time() - started:.0f}s -> {matching_path}, {candidate_path}")
        return 0 if not issues else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
