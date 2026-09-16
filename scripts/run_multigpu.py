"""
Shard a grid run across every available GPU, one worker process per GPU.

The workload is compute-bound on the reader (a single A100 sits at 100% during a
run), so throughput scales close to linearly with GPU count. Each worker holds
its own copy of the 32 GB index plus ~10 GB of models, which fits comfortably in
an 80 GB A100, so no index sharding is needed.

Sharding is BY DATASET, not by cell, because `truncate` takes its token budget
from the `filter_summ` run on the same dataset - splitting those across
processes would silently lose the budget matching.
"""
import argparse, os, subprocess, sys, time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS = ["hotpotqa", "2wiki", "musique", "nq_open", "triviaqa"]


def visible_gpus():
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env:
        return [g.strip() for g in env.split(",") if g.strip()]
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True)
        return [str(i) for i, _ in enumerate(out.stdout.strip().splitlines())]
    except Exception:
        return ["0"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--conditions", nargs="+", default=["baseline", "truncate", "filter_summ"])
    ap.add_argument("--rerank", nargs="+", default=["off", "on"])
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--allow-stale", action="store_true")
    ap.add_argument("--top-k", type=int, default=30)
    ap.add_argument("--summary-model", default="google/flan-t5-small")
    ap.add_argument("--reader", default=None)
    ap.add_argument("--out-dir", default="OUTPUT/full")
    ap.add_argument("--log-dir", default="logs/multigpu")
    ap.add_argument("--gpus", default=None,
                    help="comma-separated GPU ids; default = all visible")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the sharding plan and the per-worker commands, launch nothing")
    args = ap.parse_args()

    gpus = [g.strip() for g in args.gpus.split(",")] if args.gpus else visible_gpus()
    if not args.dry_run:
        os.makedirs(args.out_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)

    # round-robin whole datasets onto GPUs
    shards = {g: [] for g in gpus}
    for i, ds in enumerate(args.datasets):
        shards[gpus[i % len(gpus)]].append(ds)

    print(f"{len(gpus)} GPU(s): {', '.join(gpus)}")
    for g, ds in shards.items():
        print(f"  gpu {g}: {' '.join(ds) if ds else '(idle)'}")
    if len(args.datasets) < len(gpus):
        print(f"note: only {len(args.datasets)} datasets for {len(gpus)} GPUs - "
              "some will idle; split by condition instead for finer granularity")

    procs = []
    for g, ds in shards.items():
        if not ds:
            continue
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=g)
        cmd = [sys.executable, "scripts/run_grid.py",
               "--datasets", *ds,
               "--conditions", *args.conditions,
               "--rerank", *args.rerank,
               "--n", str(args.n), "--seed", str(args.seed),
               "--top-k", str(args.top_k),
               "--summary-model", args.summary_model,
               "--out-dir", args.out_dir]
        if args.reader:
            cmd += ["--reader", args.reader]
        if args.allow_stale:
            cmd += ["--allow-stale"]
        if args.dry_run:
            print(f"  [gpu {g}] " + " ".join(cmd))
            continue
        log = open(f"{args.log_dir}/gpu{g}.log", "w")
        procs.append((g, subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log, stderr=log)))
        print(f"launched gpu {g} -> {args.log_dir}/gpu{g}.log", flush=True)
        time.sleep(5)          # stagger the index loads

    if args.dry_run:
        print("dry run - nothing launched")
        return 0

    failed = 0
    for g, p in procs:
        rc = p.wait()
        print(f"gpu {g} exited rc={rc}", flush=True)
        failed += rc != 0
    print("ALL WORKERS DONE" if not failed else f"{failed} worker(s) FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
