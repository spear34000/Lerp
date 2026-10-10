"""Real-model check of the merge methods in the resident evaluator: apply time and score of linear / slerp / ties / dare_ties /
dare_linear blends of two full checkpoints (same genes), plus the two parents.

    python experiments/merge_methods_check.py --base DIR --parent a=DIR --parent b=DIR --device xpu --items 100
"""
import argparse
import time

from lerp.resident import ResidentSession
from lerp.spec import parse_spec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--parent", action="append", required=True, help="name=path (two)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--items", type=int, default=100)
    ap.add_argument("--density", type=float, default=0.5)
    ap.add_argument("--task-scale", type=float, default=1.0)
    ap.add_argument("--weight", type=float, default=0.5, help="gene value (weight of the first parent) for every group")
    ap.add_argument("--methods", nargs="+", default=["linear", "slerp", "ties", "dare_ties", "dare_linear"])
    args = ap.parse_args()
    raw = {"name": "mm", "base_model": args.base, "genes": 2, "out_dtype": "bfloat16", "gene_groups": ["attention", "mlp", "router", "other"],
           "population": 3, "density": args.density, "task_scale": args.task_scale, "seed": 5,
           "parents": [{"name": n, "model": m} for n, m in (x.split("=", 1) for x in args.parent)],
           "evaluation": {"limit": args.items, "tasks": {"arc_easy": {"metric": "acc_norm,none"}, "boolq": {"metric": "acc,none"}}}}
    spec = parse_spec(raw)
    session = ResidentSession(spec, args.device, "bfloat16")
    window = (0, args.items)
    for p in spec.parents:
        t0 = time.time()
        print(f"{p.name:<12}", session.evaluate_reference(p.name, window), f"{time.time() - t0:.0f}s", flush=True)
    genes = [args.weight] * spec.genome_size
    for method in args.methods:
        t0 = time.time()
        session.blender.apply(genes, method)
        t1 = time.time()
        print(f"{method:<12}", session._score(window, 9000), f"apply {t1 - t0:.0f}s, score {time.time() - t1:.0f}s", flush=True)
    session.close()


if __name__ == "__main__":
    main()
