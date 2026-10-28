"""
Continuous self-play loop in small generations (as in KataGo instead of a few large iterations with hard gates).

Every generation:
1. collect: SelfPlayCollector plays games with the newest net (policy target pruning and root noise as configured there)
2. convert: DatasetProcessing/selfplay_to_dataset.py turns the run into training arrays
3. train: train.py continues from the previous checkpoint for a fixed number of samples at a constant learning rate, on a
   window of the newest generations (equal shares) plus extra data (e.g. the v2 data, against the value drift)
4. export: export_onnx.py writes the net the next generation collects with

Every --eval-every generations an evaluation command (e.g. an SPRT mirror duel) compares the newest net with the one
evaluated last; its last output line is logged. The loop stops if that line reports the newest net as clearly worse
(the upper end of the Elo interval below 0), and otherwise keeps going until --generations are done or it is stopped.
The state is kept in loop_state.json in the loop directory, so a stopped loop continues where it left off.

Usage (from Agents/Training, with the Agents/env interpreter):
  python selfplay_loop.py --loop-dir DIR --start-checkpoint CKPT --start-model ONNX --extra-data DIR@0.3 [options]
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
COLLECTOR = os.path.join(REPO, "SelfPlayCollector", "bin", "Release", "net10.0", "SelfPlayCollector.exe")


def log(loop_dir, text):
    line = f"{time.strftime('%H:%M')} {text}"
    print(line, flush=True)
    with open(os.path.join(loop_dir, "loop.log"), "a", encoding="utf-8") as file:
        file.write(line + "\n")


def run(command, output_path, env=None):
    """Runs a command with its output in a file; raises with the end of the output if it fails"""
    with open(output_path, "w", encoding="utf-8") as output:
        code = subprocess.call(command, stdout=output, stderr=subprocess.STDOUT, env=env)
    if code != 0:
        with open(output_path, encoding="utf-8", errors="replace") as file:
            tail = file.read()[-2000:]
        raise RuntimeError(f"{os.path.basename(command[0])} failed with exit code {code}:\n{tail}")


def load_state(loop_dir):
    path = os.path.join(loop_dir, "loop_state.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as file:
            return json.load(file)
    return {"generations": []}


def save_state(loop_dir, state):
    path = os.path.join(loop_dir, "loop_state.json")
    with open(path + ".tmp", "w", encoding="utf-8") as file:
        json.dump(state, file, indent=2)
    os.replace(path + ".tmp", path)


def clearly_worse(line):
    """Whether an evaluation line ("... +12 Elo [-30, +55] ...") puts the upper end of the interval below 0"""
    match = re.search(r"([+-]?\d+) Elo \[([+-]?\d+), ([+-]?\d+)\]", line)
    return match is not None and int(match.group(3)) < 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Continuous self-play loop in small generations")
    parser.add_argument("--loop-dir", required=True)
    parser.add_argument("--start-checkpoint", required=True, help="checkpoint of the net generation 1 starts from")
    parser.add_argument("--start-model", required=True, help="its ONNX export, which collects generation 1")
    parser.add_argument("--generations", type=int, default=1000)
    parser.add_argument("--games", type=int, default=8000, help="games per generation")
    parser.add_argument("--collector-args", default="--concurrent 1024 --nets 3", help="further SelfPlayCollector options")
    parser.add_argument("--seed", type=int, default=20261100, help="generation g collects with seed + g")
    parser.add_argument("--window", type=int, default=4, help="newest generations in the training data")
    parser.add_argument("--extra-data", action="append", default=[], help="further prepared data DIR@SHARE, e.g. the v2 data")
    parser.add_argument("--samples", type=int, default=2_000_000, help="training samples per generation")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--target-lambda", type=float, default=0.5)
    parser.add_argument("--policy-temperature", type=float, default=2.0)
    parser.add_argument("--train-args", default="--normalize-inputs --placement-weight 0.5 --vp-share-weight 0.5 --rounds-weight 0.2 "
                                                  "--augment-rotation --augment-board --policy-weight 1.0")
    parser.add_argument("--eval-every", type=int, default=3)
    parser.add_argument("--eval-command", default=None,
                        help="command evaluating {new} (ONNX) against {old} (ONNX), {name} names the run; its last output line is logged")
    args = parser.parse_args(argv)

    os.makedirs(args.loop_dir, exist_ok=True)
    state = load_state(args.loop_dir)
    python = sys.executable
    env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE")
    extra_shares = sum(float(spec.rsplit("@", 1)[1]) for spec in args.extra_data)
    if not 0 <= extra_shares < 1:
        raise SystemExit("The shares of --extra-data must add up to less than 1")

    def newest(key, fallback):
        done = [g for g in state["generations"] if key in g]
        return done[-1][key] if done else fallback

    while len([g for g in state["generations"] if "model" in g]) < args.generations:
        if state["generations"] and "model" not in state["generations"][-1]:
            gen = state["generations"][-1]
        else:
            gen = {"generation": len(state["generations"]) + 1}
            state["generations"].append(gen)
            save_state(args.loop_dir, state)
        g = gen["generation"]
        gen_dir = os.path.join(args.loop_dir, f"gen{g:03d}")
        os.makedirs(gen_dir, exist_ok=True)
        model = newest("model", args.start_model)
        checkpoint = newest("checkpoint", args.start_checkpoint)

        if "run" not in gen:
            # A collection stopped before it finished left its directory behind; the collector won't reuse it
            unfinished = os.path.join(gen_dir, "run")
            if os.path.exists(unfinished):
                os.replace(unfinished, os.path.join(gen_dir, f"run_unfinished_{time.strftime('%Y%m%d_%H%M%S')}"))
            started = time.time()
            command = [COLLECTOR, "--games", str(args.games), "--output", gen_dir, "--name", "run", "--seed", str(args.seed + g),
                       "--model", model] + args.collector_args.split()
            run(command, os.path.join(gen_dir, "collect.log"))
            with open(os.path.join(gen_dir, "run", "metadata.json"), encoding="utf-8") as file:
                result = json.load(file)["result"]
            gen["run"] = os.path.join(gen_dir, "run")
            gen["collect"] = {"games": result["games"], "positions": result["positions"], "errors": result["errors"],
                              "minutes": round((time.time() - started) / 60, 1), "model": model}
            save_state(args.loop_dir, state)
            log(args.loop_dir, f"gen {g}: {result['games']} games, {result['positions']} positions, {result['errors']} errors "
                               f"in {gen['collect']['minutes']} min")

        if "prepared" not in gen:
            prepared = os.path.join(gen_dir, "prepared")
            run([python, os.path.join(REPO, "DatasetProcessing", "selfplay_to_dataset.py"), gen["run"], "--output", prepared,
                 "--target-lambda", str(args.target_lambda)], os.path.join(gen_dir, "convert.log"), env)
            gen["prepared"] = prepared
            save_state(args.loop_dir, state)

        if "checkpoint" not in gen:
            window = [x["prepared"] for x in state["generations"] if "prepared" in x][-args.window:]
            share = (1 - extra_shares) / len(window)
            # The newest generation first: the validation runs on its held-out games
            data = [f"{path}@{share}" for path in reversed(window)] + args.extra_data
            checkpoint_dir = os.path.join(gen_dir, "checkpoints")
            command = [python, os.path.join(HERE, "train.py"), "--init-from", checkpoint, "--target-lambda", str(args.target_lambda),
                       "--policy-temperature", str(args.policy_temperature), "--lr", str(args.lr), "--constant-lr",
                       "--max-epochs", "1", "--epoch-samples", str(args.samples), "--checkpoint-dir", checkpoint_dir]
            for spec in data:
                command += ["--data", spec]
            started = time.time()
            run(command + args.train_args.split(), os.path.join(gen_dir, "train.log"), env)
            checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "*.ckpt")), key=os.path.getmtime)
            if not checkpoints:
                raise RuntimeError(f"train.py wrote no checkpoint into {checkpoint_dir}")
            gen["checkpoint"] = checkpoints[-1]
            gen["train"] = {"minutes": round((time.time() - started) / 60, 1), "window": len(window),
                            "val_loss": re.search(r"val_loss=([0-9]+\.[0-9]+)", os.path.basename(checkpoints[-1])).group(1)}
            save_state(args.loop_dir, state)

        if "model" not in gen:
            onnx = os.path.join(gen_dir, f"state_value_net-loop-g{g:03d}.onnx")
            run([python, os.path.join(HERE, "export_onnx.py"), "--ckpt", gen["checkpoint"], "--out", onnx],
                os.path.join(gen_dir, "export.log"), env)
            gen["model"] = onnx
            save_state(args.loop_dir, state)
            log(args.loop_dir, f"gen {g}: trained on {gen['train']['window']} generations in {gen['train']['minutes']} min, "
                               f"val_loss {gen['train']['val_loss']}, net {os.path.basename(onnx)}")

        if args.eval_command and g % args.eval_every == 0 and "eval" not in gen:
            reference = state.get("evaluated", args.start_model)
            command = args.eval_command.format(new=gen["model"], old=reference, name=f"loop_g{g:03d}")
            output = os.path.join(gen_dir, "eval.log")
            with open(output, "w", encoding="utf-8") as file:
                subprocess.call(command, shell=True, stdout=file, stderr=subprocess.STDOUT)
            with open(output, encoding="utf-8", errors="replace") as file:
                lines = [line.strip() for line in file if line.strip()]
            gen["eval"] = {"against": reference, "result": lines[-1] if lines else "no output"}
            state["evaluated"] = gen["model"]
            save_state(args.loop_dir, state)
            log(args.loop_dir, f"gen {g} against {os.path.basename(reference)}: {gen['eval']['result']}")
            if clearly_worse(gen["eval"]["result"]):
                log(args.loop_dir, f"gen {g} is clearly worse than {os.path.basename(reference)}, the loop stops")
                return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
