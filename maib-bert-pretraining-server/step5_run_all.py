"""
STEP 5 - Fine-tune every configuration of config.FINETUNE_CONFIGS with every encoder and every seed.

Encoders: the original BERT ("base") and every adapted model in models/ (step 3).
A run whose result file already exists is skipped, so the script can be stopped and started again; an
interrupted run continues after its last finished epoch (step 4).

  python step5_run_all.py                                     # base + all models in models/
  python step5_run_all.py --encoders models/bert-maib-external
  python step5_run_all.py --skip-base                         # base results copied from the paper (see README)

On a CPU server with many cores (no GPU), run several fine-tunings at the same time:
  python step5_run_all.py --workers 4 --threads 20            # 4 runs at once, 20 cores each (80 cores)
One PyTorch process does not use 80 cores efficiently; several processes with a share of the cores each finish
the 20 runs much sooner. Each run is a separate process (step4_finetune.py) and writes its own log to logs/.
Every run is trained exactly as when run alone: same seed, same data order, same settings.
"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import config
from step4_finetune import run_name, train

parser = argparse.ArgumentParser()
parser.add_argument("--encoders", nargs="+", help="encoders to fine-tune (default: base + every folder in models/)")
parser.add_argument("--skip-base", action="store_true", help="do not fine-tune the original BERT")
parser.add_argument("--workers", type=int, default=1, help="fine-tunings at the same time (default 1, one after the other)")
parser.add_argument("--threads", type=int, help="CPU cores for each fine-tuning when --workers > 1 "
                                                "(default: all cores divided by --workers)")
args = parser.parse_args()

encoders = args.encoders or sorted(str(p) for p in Path(config.MODELS_DIR).glob("*") if (p / "config.json").exists())
if not args.skip_base and config.BASE_MODEL not in encoders:
    encoders = [config.BASE_MODEL] + encoders
jobs = [(e, m, l, s) for e in encoders for m, l in config.FINETUNE_CONFIGS for s in config.SEEDS]
print(f"{len(jobs)} runs: {len(encoders)} encoders x {len(config.FINETUNE_CONFIGS)} configurations "
      f"x {len(config.SEEDS)} seeds")


def done(encoder, model, loss, seed):
    """A finished run has a result file. For the original BERT, a file copied from the paper (named without
    the "base__" prefix, e.g. bert__ce__seed1.json) counts too, so those runs are not trained again."""
    results = Path(config.RESULTS_DIR)
    if (results / f"{run_name(encoder, model, loss, seed)}.json").exists():
        return True
    return encoder == config.BASE_MODEL and (results / f"{model}__{loss}__seed{seed}.json").exists()


todo = []
for i, job in enumerate(jobs, start=1):
    if done(*job):
        print(f"[{i}/{len(jobs)}] {run_name(*job)}: already done, skipped")
    else:
        todo.append(job)

# ---------------------------------------------------------------------------- one after the other (GPU)
if args.workers <= 1:
    for k, (encoder, model, loss, seed) in enumerate(todo, start=1):
        print(f"[{k}/{len(todo)} to do]", end="")
        train(model, loss, seed, encoder=encoder)
    sys.exit(0)

# ---------------------------------------------------------------------------- several at once (CPU server)
threads = args.threads or max(1, (os.cpu_count() or 1) // args.workers)
logs = Path("logs")
logs.mkdir(exist_ok=True)
print(f"{len(todo)} runs to do, {args.workers} at a time with {threads} CPU threads each; "
      f"progress of each run in {logs}/<run>.log")


def run_one(job):
    encoder, model, loss, seed = job
    name = run_name(*job)
    # every library that can start threads gets the same limit, so the processes do not fight over the cores
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
               OPENBLAS_NUM_THREADS=str(threads), TOKENIZERS_PARALLELISM="false")
    start = time.time()
    with open(logs / f"{name}.log", "a") as log:
        process = subprocess.Popen([sys.executable, "step4_finetune.py", "--model", model, "--loss", loss,
                                    "--seed", str(seed), "--encoder", encoder], stdout=log,
                                   stderr=subprocess.STDOUT, env=env)
        running.append(process)
        code = process.wait()
    return name, code, (time.time() - start) / 60


failed, running = [], []
pool = ThreadPoolExecutor(max_workers=args.workers)
futures = [pool.submit(run_one, job) for job in todo]
try:
    for k, future in enumerate(as_completed(futures), start=1):
        name, code, minutes = future.result()
        if code == 0:
            last = [l for l in (logs / f"{name}.log").read_text().splitlines() if l.startswith("[test]")]
            print(f"[{k}/{len(todo)} finished] {name} in {minutes:.0f} min  {last[-1] if last else ''}", flush=True)
        else:
            failed.append(name)
            print(f"[{k}/{len(todo)} FAILED] {name} (exit code {code}); see {logs}/{name}.log", flush=True)
except KeyboardInterrupt:              # Ctrl+C or a time limit: stop every running fine-tuning, then exit
    for future in futures:
        future.cancel()                # runs that have not started yet
    for process in running:
        process.kill()                 # each continues next time after its last finished epoch
    pool.shutdown(wait=False, cancel_futures=True)
    sys.exit("Stopped. Run the same command again to continue.")
pool.shutdown()
if failed:
    sys.exit(f"{len(failed)} run(s) failed: {', '.join(failed)}. Fix the cause and run again: "
             "finished runs are skipped, stopped ones continue after their last finished epoch.")
