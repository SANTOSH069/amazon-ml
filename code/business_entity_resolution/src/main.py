import argparse
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from io_utils import find_data_dir, load_split, profile, validate, write_lists
from resolver import Resolver


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None,
                    help="folder containing train/ and test/ (auto-detected if omitted)")
    ap.add_argument("--out-dir", default=None, help="default: output/ next to the dataset")
    ap.add_argument("--model", default=None, help="default: model/resolver.pkl in the project")
    ap.add_argument("--mode", choices=["all", "fit", "predict"], default="all")
    ap.add_argument("--k", type=int, default=5, help="stage-1 Source 1 candidates per target")
    ap.add_argument("--target-recall", type=float, default=0.995,
                    help="share of stage-1 true pairs the stage-2 filter must keep")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
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
        s1, s2, s3, gt = load_split(data_dir, "train")
        if gt is None:
            print("train_ground_truth.tsv not found - it is required for training.")
            return 2
        profile("train", s1, s2, s3, gt)
        res = Resolver(k=args.k, target_recall=args.target_recall, seed=args.seed)
        res.fit(s1, s2, s3, gt)
        res.log = None
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as fh:
            pickle.dump(res, fh, protocol=5)
        print(f"[fit] saved {model_path}  ({time.time() - t0:.0f}s)", flush=True)
        del s1, s2, s3, gt

    if args.mode in ("all", "predict"):
        with open(model_path, "rb") as fh:
            res = pickle.load(fh)
        res.log = lambda *a: print(*a, flush=True)
        s1, s2, s3, _ = load_split(data_dir, "test")
        profile("test", s1, s2, s3, None)
        cands, matches = res.predict(s1, s2, s3)
        m_path = os.path.join(out_dir, "matching_results.tsv")
        c_path = os.path.join(out_dir, "candidate_pairs.tsv")
        ids = s1.entity_id.tolist()
        write_lists(m_path, ids, matches, "matched_entity_ids")
        write_lists(c_path, ids, cands, "candidate_entity_ids")
        issues = validate(m_path, c_path, s1, s2, s3)
        n_m = sum(len(v) for v in matches.values())
        n_c = sum(len(v) for v in cands.values())
        print(f"[test] S1={len(ids)} candidates={n_c} ({n_c / max(len(ids), 1):.2f}/S1) "
              f"matches={n_m} ({n_m / max(len(ids), 1):.2f}/S1) "
              f"non-empty={sum(bool(v) for v in matches.values())}")
        print("[validate] PASS" if not issues else "[validate] FAIL\n  " + "\n  ".join(issues))
        print(f"done in {time.time() - t0:.0f}s -> {m_path}, {c_path}")
        return 0 if not issues else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
