#!/usr/bin/env python3
"""Check training progress from TensorBoard logs."""

import argparse
import glob
import os
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_logs(log_dir="logs"):
    """Load TensorBoard events from log directory."""
    # Check for subdirectories first, then flat event files
    dirs = sorted(glob.glob(os.path.join(log_dir, "*/")))
    event_files = sorted(glob.glob(os.path.join(log_dir, "events.out.tfevents.*")))

    if dirs:
        path = dirs[-1]
    elif event_files:
        path = log_dir
    else:
        print(f"No TensorBoard logs found in {log_dir}/")
        sys.exit(1)

    ea = EventAccumulator(path, size_guidance={"scalars": 0})  # load all
    ea.Reload()
    return path, ea


def get_latest(ea, tag):
    """Get latest value for a tag, or None."""
    scalars = ea.Tags().get("scalars", [])
    if tag not in scalars:
        return None
    events = ea.Scalars(tag)
    return events[-1] if events else None


def get_history(ea, tag, n=None):
    """Get scalar history, optionally last n entries."""
    scalars = ea.Tags().get("scalars", [])
    if tag not in scalars:
        return []
    events = ea.Scalars(tag)
    if n is not None:
        return events[-n:]
    return events


def main():
    parser = argparse.ArgumentParser(description="Check training progress")
    parser.add_argument("--log-dir", default="logs", help="TensorBoard log directory")
    parser.add_argument("-n", type=int, default=15, help="Number of recent epochs to show")
    parser.add_argument("--all", action="store_true", help="Show all available tags")
    args = parser.parse_args()

    path, ea = load_logs(args.log_dir)
    scalars = ea.Tags().get("scalars", [])

    # Determine total epochs
    total_tag = "train_epoch/total"
    total_events = get_history(ea, total_tag)
    total_epochs = len(total_events)

    print(f"Log dir: {path}")
    print(f"Epochs completed: {total_epochs}")
    print()

    # Latest snapshot
    print("=== Latest Values ===")
    snapshot_tags = [
        ("train_epoch/total", "Train loss"),
        ("val/total", "Val loss"),
        ("train_epoch/delay", "Train delay loss"),
        ("val/delay", "Val delay loss"),
        ("train_epoch/delay_acc", "Train delay acc"),
        ("val/delay_acc", "Val delay acc"),
        ("val/erle_db", "Val ERLE (dB)"),
        ("train_epoch/plcmse", "Train PLCMSE"),
        ("train_epoch/mag_l1", "Train Mag-L1"),
        ("train_epoch/time_l1", "Train Time-L1"),
        ("train_epoch/entropy", "Train entropy"),
        ("val/plcmse", "Val PLCMSE"),
        ("val/mag_l1", "Val Mag-L1"),
        ("val/time_l1", "Val Time-L1"),
        ("val/entropy", "Val entropy"),
        ("train/temperature", "Temperature"),
        ("train/lr", "Learning rate"),
    ]
    for tag, label in snapshot_tags:
        e = get_latest(ea, tag)
        if e is not None:
            print(f"  {label:20s} {e.value:>12.6f}  (step {e.step})")
    print()

    # Epoch history table
    print(f"=== Epoch History (last {args.n}) ===")
    train_total = get_history(ea, "train_epoch/total", args.n)
    val_total = get_history(ea, "val/total", args.n)

    if train_total:
        # Build lookup dicts for extra columns
        val_by_step = {e.step: e.value for e in get_history(ea, "val/total")}
        dacc_events = {e.step: e.value for e in get_history(ea, "val/delay_acc")}
        erle_events = {e.step: e.value for e in get_history(ea, "val/erle_db")}
        temp_events = {e.step: e.value for e in get_history(ea, "train/temperature")}

        # Find best val
        all_val = get_history(ea, "val/total")
        best_val_step = min(all_val, key=lambda e: e.value).step if all_val else -1

        hdr = f"  {'Epoch':>6s}  {'Train':>10s}  {'Val':>10s}  {'DelAcc':>7s}  {'ERLE':>7s}  {'Temp':>6s}"
        print(hdr)
        print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*7}  {'-'*7}  {'-'*6}")

        for t in train_total:
            ep = t.step
            val_v = val_by_step.get(ep, None)
            dacc_v = dacc_events.get(ep, None)
            erle_v = erle_events.get(ep, None)
            temp_v = temp_events.get(ep, None)

            val_str = f"{val_v:10.4f}" if val_v is not None else "       N/A"
            dacc_str = f"{dacc_v:6.1%}" if dacc_v is not None else "    N/A"
            erle_str = f"{erle_v:+6.1f}" if erle_v is not None else "    N/A"
            temp_str = f"{temp_v:.3f}" if temp_v is not None else "   N/A"
            marker = " *" if ep == best_val_step else ""
            print(f"  {ep:>6d}  {t.value:>10.4f}  {val_str}  {dacc_str:>7s}  {erle_str:>7s}  {temp_str:>6s}{marker}")

        if all_val:
            best = min(all_val, key=lambda e: e.value)
            print(f"\n  Best val loss: {best.value:.4f} at epoch {best.step}")
    print()

    # Gradient norm summary
    grad_events = get_history(ea, "train/grad_norm", args.n * 50)
    if grad_events:
        import numpy as np
        vals = [e.value for e in grad_events[-500:]]
        clipped = sum(1 for v in vals if v >= 4.99)
        print(f"=== Gradient Norms (last {len(vals)} steps) ===")
        print(f"  Mean: {np.mean(vals):.2f}  Median: {np.median(vals):.2f}  "
              f"Max: {np.max(vals):.2f}  Clipped: {clipped}/{len(vals)}")
    print()

    # All tags
    if args.all:
        print("=== All Tags ===")
        for tag in sorted(scalars):
            events = ea.Scalars(tag)
            if events:
                print(f"  {tag:45s} ({len(events):>6d} pts, latest={events[-1].value:.4f})")


if __name__ == "__main__":
    main()
