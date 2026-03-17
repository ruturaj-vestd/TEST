from __future__ import annotations

import argparse
import time
from datetime import datetime

from stock_bot.workflow import run_daily


def run_once(output_dir: str) -> None:
    path = run_daily(output_dir=output_dir)
    print(f"Report written to {path}")


def run_scheduler(output_dir: str, run_time_utc: str) -> None:
    print(f"Scheduler started. Will run every day at {run_time_utc} UTC")
    while True:
        now = datetime.utcnow().strftime("%H:%M")
        if now == run_time_utc:
            try:
                path = run_daily(output_dir=output_dir)
                print(f"[{datetime.utcnow().isoformat()}] Report written to {path}")
                time.sleep(65)
            except Exception as exc:
                print(f"Run failed: {exc}")
                time.sleep(120)
        time.sleep(20)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-agent daily stock predictor")
    parser.add_argument("--output-dir", default="reports", help="Directory where markdown reports are written")
    parser.add_argument("--schedule", action="store_true", help="Run continuously and produce one report per day")
    parser.add_argument("--run-time-utc", default="12:30", help="UTC time (HH:MM) for daily run in scheduler mode")
    args = parser.parse_args()

    if args.schedule:
        run_scheduler(output_dir=args.output_dir, run_time_utc=args.run_time_utc)
    else:
        run_once(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
