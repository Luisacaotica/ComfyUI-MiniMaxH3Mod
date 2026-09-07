"""Benchmark the three production multi-ref strategies; reject parity failures.
Run: python gauntlet_harness.py --device cuda --refs 4 --frames 8 --edge 64
Run regressions: python -m unittest discover -s tests -v
"""
import argparse
import gc
import json
import time
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import optimize_latent_multi


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--refs", type=int, default=4)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--edge", type=int, default=64)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--output", default="tests/gauntlet_result.json")
    args = parser.parse_args()
    torch.manual_seed(123)
    initial = torch.randn(1,24,2,8,8)
    targets = [torch.randn(1,24,args.frames,args.edge,args.edge) for _ in range(args.refs)]
    # Warm up optimizer and device initialization outside the measurements.
    optimize_latent_multi(initial, targets[:1], steps=1, device=args.device, strategy="stream")
    results, baseline = [], None
    for strategy in ("resident", "stream", "grouped"):
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        start = time.perf_counter()
        value = optimize_latent_multi(initial, targets, steps=args.steps, device=args.device, strategy=strategy)
        if args.device == "cuda": torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated()/1024**2 if args.device == "cuda" else None
        value = value.cpu()
        if baseline is None: baseline = value
        error = (value-baseline).abs().max().item()
        torch.testing.assert_close(value,baseline,rtol=1e-4,atol=1e-5)
        results.append({"strategy":strategy,"seconds":elapsed,"peak_cuda_mib":peak,"max_error":error,"parity":True})
        del value
    Path(args.output).write_text(json.dumps({"parameters":vars(args),"results":results},indent=2),encoding="utf-8")
    print(json.dumps(results,indent=2))


if __name__ == "__main__":
    run()
