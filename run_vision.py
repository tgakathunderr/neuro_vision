"""Main execution script for BIB-2 Neuro-Vision: Object Recognition & Generalization."""

import argparse
import os
import sys
import time
from typing import Dict, Any, Optional
import numpy as np

# Ensure root paths are available
_BIM2_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_BIB2_PATH = os.path.join(_BIM2_ROOT, "BIB-2")
if _BIB2_PATH not in sys.path:
    sys.path.insert(0, _BIB2_PATH)
if _BIM2_ROOT not in sys.path:
    sys.path.insert(0, _BIM2_ROOT)

from bib2.brain import BIB2NervousSystem
from neuro_vision.adapter import VisionAdapter
from neuro_vision.dataset import generate_benchmark_dataset, CLASSES
from neuro_vision.hud import VisionHUD

DEFAULT_CKPT = os.path.join(os.path.dirname(__file__), "models", "vision_brain.npz")


def run_benchmark(
    adapter: VisionAdapter,
    dataset: Dict[str, Any],
    verbose: bool = True,
    use_preprocessing: bool = True,
) -> Dict[str, Any]:
    """
    Executes the standardized visual object recognition benchmark across all 30 unseen test images:
    - 6 semantic classes (Cat, Car, Airplane, Tree, Coffee Cup, Star)
    - 5 transformation variations per class: Rotated, Scaled, Translated, Noisy, Alternate Exemplar
    """
    test_samples = dataset["test_samples"]
    total = len(test_samples)
    correct = 0
    by_variation: Dict[str, List[int]] = {}  # var -> [correct, total]

    mode_label = "FULL PIPELINE (FOVEATED & SCALED)" if use_preprocessing else "RAW VENTRAL STREAM (NO PREPROCESSING)"

    if verbose:
        print("\n=========================================================================")
        print(f" BIB-2 NEURO-VISION: EMPIRICAL BENCHMARK [{mode_label}]")
        print("=========================================================================")
        print(f"{'#':<3} | {'True Class':<12} | {'Variation':<14} | {'Predicted':<12} | {'Conf':<6} | {'Status'}")
        print("-" * 73)

    for idx, sample in enumerate(test_samples):
        res = adapter.predict(sample["image"], use_preprocessing=use_preprocessing)
        pred_cls = res["predicted_class"]
        conf = res["confidence"]
        true_cls = sample["class"]
        var_name = sample["variation"]
        is_corr = (pred_cls == true_cls)

        if is_corr:
            correct += 1

        if var_name not in by_variation:
            by_variation[var_name] = [0, 0]
        by_variation[var_name][1] += 1
        if is_corr:
            by_variation[var_name][0] += 1

        if verbose:
            status_tag = "[MATCH]" if is_corr else "[MISMATCH]"
            print(f"{idx + 1:<3} | {true_cls:<12} | {var_name:<14} | {pred_cls:<12} | {conf * 100:.1f}% | {status_tag}")

    acc_pct = (correct / total * 100.0) if total > 0 else 0.0

    if verbose:
        print("=" * 73)
        print(f" OVERALL GENERALIZATION ACCURACY: {correct}/{total} ({acc_pct:.1f}%)")
        print(" ACCURACY BY TRANSFORMATION TYPE:")
        for var_name, (c_cnt, t_cnt) in by_variation.items():
            v_pct = (c_cnt / t_cnt * 100.0) if t_cnt > 0 else 0.0
            bar = "#" * int(v_pct / 10) + "-" * (10 - int(v_pct / 10))
            print(f"   * {var_name.capitalize():<12}: {c_cnt}/{t_cnt} ({v_pct:.1f}%)  [{bar}]")
        print("=========================================================================\n")

    return {
        "correct": correct,
        "total": total,
        "accuracy_pct": acc_pct,
        "by_variation": by_variation,
        "use_preprocessing": use_preprocessing,
    }


def run_ablation_benchmark(adapter: VisionAdapter, dataset: Dict[str, Any]) -> Dict[str, Any]:
    """
    Rigorously tests and separates CA3 neural learning from algorithmic foveation preprocessing.
    Runs:
    1. Condition A: Raw Ventral Stream (Centroid shift & size constancy DISABLED).
    2. Condition B: Full Biological Pipeline (With Saccadic Foveation & Size Constancy).
    """
    print("\n" + "=" * 80)
    print(" BIB-2 NEURO-VISION: EMPIRICAL ABLATION & GENERALIZATION ATTRIBUTION")
    print("=" * 80)

    # 1. Raw Ventral Stream
    raw_res = run_benchmark(adapter, dataset, verbose=False, use_preprocessing=False)

    # 2. Full Pipeline
    full_res = run_benchmark(adapter, dataset, verbose=False, use_preprocessing=True)

    print(f"\n{'Transformation':<16} | {'Raw Ventral (No Preproc)':<26} | {'Full Pipeline':<15} | {'Mechanistic Source'}")
    print("-" * 80)

    variations = list(full_res["by_variation"].keys())
    for var in variations:
        r_corr, r_tot = raw_res["by_variation"].get(var, [0, 6])
        f_corr, f_tot = full_res["by_variation"].get(var, [0, 6])
        r_pct = (r_corr / r_tot * 100.0) if r_tot > 0 else 0.0
        f_pct = (f_corr / f_tot * 100.0) if f_tot > 0 else 0.0

        if var in ["rotated", "alternate_exemplar"]:
            source = "Pure CA3 Learning & Invariance"
        elif var == "noisy":
            source = "CA3 Completion + TRN Filter"
        else:
            source = "Saccadic/Cortical Preproc"

        raw_str = f"{r_corr}/{r_tot} ({r_pct:5.1f}%)"
        full_str = f"{f_corr}/{f_tot} ({f_pct:5.1f}%)"
        print(f"{var.capitalize():<16} | {raw_str:<26} | {full_str:<15} | {source}")

    print("-" * 80)
    print(
        f"{'OVERALL TOTAL':<16} | "
        f"{raw_res['correct']}/{raw_res['total']} ({raw_res['accuracy_pct']:5.1f}%)"
        f"{' ' * 11} | "
        f"{full_res['correct']}/{full_res['total']} ({full_res['accuracy_pct']:5.1f}%)"
        f"{' ' * 2} | "
        f"Honest Decomposition"
    )
    print("=" * 80 + "\n")

    return {
        "raw": raw_res,
        "full": full_res,
    }


def interactive_session(adapter: VisionAdapter, dataset: Dict[str, Any], headless: bool = False) -> None:
    """Runs interactive cybernetic HUD."""
    import pygame

    hud = VisionHUD()
    test_samples = dataset["test_samples"]
    current_idx = 0
    benchmark_results: Optional[Dict[str, Any]] = None

    running = True
    while running:
        sample = test_samples[current_idx]
        pred_result = adapter.predict(sample["image"])

        hud.render(
            current_sample=sample,
            prediction_result=pred_result,
            benchmark_stats=benchmark_results,
            sample_idx=current_idx,
            total_samples=len(test_samples),
        )

        if headless:
            # Single frame render test for headless CI
            break

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False

                elif event.key in (pygame.K_SPACE, pygame.K_RIGHT):
                    current_idx = (current_idx + 1) % len(test_samples)

                elif event.key in (pygame.K_p, pygame.K_LEFT):
                    current_idx = (current_idx - 1) % len(test_samples)

                elif event.key == pygame.K_t:
                    print("[INFO] Re-training 1-shot prototype memories into Hippocampus CA3...")
                    adapter.train_prototypes(dataset["prototypes"], include_saccades=True)

                elif event.key == pygame.K_a:
                    print("[INFO] Running automated empirical benchmark...")
                    benchmark_results = run_benchmark(adapter, dataset, verbose=True)

                elif event.key == pygame.K_s:
                    adapter.save_checkpoint(DEFAULT_CKPT)
                    print(f"[INFO] Checkpoint saved successfully to {DEFAULT_CKPT}")

    pygame.quit()


def main():
    parser = argparse.ArgumentParser(description="BIB-2 Neuro-Vision: Biomimetic Object Recognition & Generalization")
    parser.add_argument("--benchmark", action="store_true", help="Run automated 30-sample benchmark with ablation analysis")
    parser.add_argument("--ablation", action="store_true", help="Run comparative ablation benchmark (Raw Ventral vs Foveated)")
    parser.add_argument("--train", action="store_true", help="Train 1-shot prototypes into Hippocampal CA3 and save model")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CKPT, help="Path to checkpoint file (.npz)")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode without GUI window")
    args = parser.parse_args()

    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    print("[*] Initializing BIB-2 Human Nervous System Container (feature_dim=64)...")
    brain = BIB2NervousSystem(feature_dim=64, seed=42)
    adapter = VisionAdapter(brain)

    print("[*] Generating procedural dataset: 6 classes x (1 prototype + 5 unseen novel transformations)...")
    dataset = generate_benchmark_dataset()

    # If checkpoint exists, load it; otherwise train 1-shot prototypes
    if os.path.exists(args.checkpoint) and not args.train:
        print(f"[*] Loading pre-trained vision brain from {args.checkpoint}...")
        adapter.load_checkpoint(args.checkpoint)
    else:
        print("[*] Executing 1-Shot Episodic Imprinting into Hippocampal CA3 & Basal Ganglia D1...")
        train_stats = adapter.train_prototypes(dataset["prototypes"], include_saccades=True)
        print(f"[*] Imprinted {train_stats['total_imprinted']} attractor patterns across {len(CLASSES)} classes.")
        print(f"[*] SWS Sleep Consolidation: {train_stats['sleep_stage']}")
        adapter.save_checkpoint(args.checkpoint)
        print(f"[*] Model checkpoint persisted to {args.checkpoint}")

    if args.benchmark or args.ablation:
        run_ablation_benchmark(adapter, dataset)
    else:
        # Pre-compute benchmark stats for HUD scorecard
        bench_stats = run_benchmark(adapter, dataset, verbose=False, use_preprocessing=True)
        print(f"[*] Benchmark Ready: {bench_stats['correct']}/{bench_stats['total']} ({bench_stats['accuracy_pct']:.1f}%)")
        print("[*] Launching Cybernetic Vision HUD...")
        interactive_session(adapter, dataset, headless=args.headless)


if __name__ == "__main__":
    main()
