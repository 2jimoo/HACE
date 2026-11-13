"""
run_phases.py — two-phase ReasoningBank experiment over a goal dataset.

Phase 1 (evolve): on the train goals, attacker and defender both learn. After each run the
  trajectory is distilled into attack and defense memory items, and both sides already use what
  the banks hold so far (attacker: --use-memory, target: --defense-memory on).
Phase 2 (adapt): on held-out test goals (similar but new attacks), the banks are frozen and the
  target retrieves defense items for each incoming attack prompt by attack similarity.
  With --baseline, the test goals are also run without defense memory to compare attack success rates.

Options not listed here are passed through to main.py, e.g.
  python run_phases.py --limit 10 --attack-model ollama/llama3.2:3b --target-model ollama/llama3.2:3b \\
      --judge-model ollama/llama-guard3:8b --no-wandb -v
"""

import os
import json
import time
import random
import argparse
import pandas as pd
import main as pair
from loggers import logger

HERE = os.path.dirname(os.path.abspath(__file__))


def run_goal(row, index, base_argv, phase_argv, phase, out_f):
    argv = base_argv + ["--goal", row["goal"], "--target-str", row["target"],
                        "--category", str(row["category"]), "--index", str(index)] + phase_argv
    args = pair.parse_args(argv)
    try:
        result = pair.main(args)
    except Exception as e:  # one failed goal (e.g. attacker never produced valid JSON) should not end the experiment
        logger.error(f"[{phase}] goal {index} failed: {e}")
        result = dict(goal=row["goal"], category=row["category"], error=str(e))
    result.update(phase=phase, index=index)
    out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
    out_f.flush()
    return result


def summarize(results):
    done = [r for r in results if "error" not in r]
    n_jb = sum(r["jailbroken"] for r in done)
    return dict(n_goals=len(results), n_errors=len(results) - len(done), n_jailbroken=n_jb,
                attack_success_rate=n_jb / len(done) if done else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=os.path.join(HERE, "data", "harmful_behaviors_custom.csv"),
                        help="CSV with goal, target, category columns.")
    parser.add_argument("--limit", type=int, default=None, help="Use only the first N goals.")
    parser.add_argument("--test-frac", type=float, default=0.3, help="Fraction of goals held out for Phase 2.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the train/test split.")
    parser.add_argument("--phases", default="1,2", choices=["1", "2", "1,2"], metavar="PHASES",
                        help="Phases to run: 1, 2 or 1,2. '2' alone evaluates against existing banks in --out-dir (same --seed).")
    parser.add_argument("--baseline", action="store_true", help="Also run Phase 2 goals without defense memory.")
    parser.add_argument("--out-dir", default=None, help="Results and memory banks go here. Default: results/phases-<timestamp>.")
    parser.add_argument("--attack-bank", default=None, help="Attack memory bank. Default: <out-dir>/attack_bank.jsonl.")
    parser.add_argument("--defense-bank", default=None, help="Defense memory bank. Default: <out-dir>/defense_bank.jsonl.")
    args, passthrough = parser.parse_known_args()

    out_dir = args.out_dir or os.path.join(HERE, "results", time.strftime("phases-%Y%m%d-%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    attack_bank = args.attack_bank or os.path.join(out_dir, "attack_bank.jsonl")
    defense_bank = args.defense_bank or os.path.join(out_dir, "defense_bank.jsonl")

    df = pd.read_csv(args.data)
    if args.limit:
        df = df.head(args.limit)
    indices = list(range(len(df)))
    random.Random(args.seed).shuffle(indices)
    n_test = max(1, round(len(indices) * args.test_frac))
    test_idx, train_idx = sorted(indices[:n_test]), sorted(indices[n_test:])
    with open(os.path.join(out_dir, "split.json"), "w") as f:
        json.dump(dict(train=train_idx, test=test_idx, seed=args.seed, data=args.data), f, indent=2)

    memory = ["--memory-path", attack_bank, "--use-memory", "--defense-memory-path", defense_bank]
    phases = []
    if "1" in args.phases:
        phases.append(("phase1", train_idx, memory + ["--defense-memory", "on"]))
    if "2" in args.phases:
        phases.append(("phase2", test_idx, memory + ["--defense-memory", "on", "--freeze-memory"]))
        if args.baseline:
            phases.append(("phase2_baseline", test_idx, memory + ["--defense-memory", "off", "--freeze-memory"]))

    summary = {}
    with open(os.path.join(out_dir, "results.jsonl"), "a", encoding="utf-8") as out_f:
        for phase, idx, phase_argv in phases:
            print(f"\n===== {phase}: {len(idx)} goals =====")
            results = []
            for k, i in enumerate(idx, 1):
                print(f"[{phase}] {k}/{len(idx)} goal {i}: {df.iloc[i]['goal']}")
                results.append(run_goal(df.iloc[i], i, passthrough, phase_argv, phase, out_f))
            summary[phase] = summarize(results)
            print(f"[{phase}] {summary[phase]}")

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary: {json.dumps(summary, indent=2)}\nResults in {out_dir}")


if __name__ == "__main__":
    main()
