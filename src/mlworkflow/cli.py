"""`mlwf` command line: train, drift, traffic, schedule."""

from __future__ import annotations

import argparse
import json

from mlworkflow.config import DEFAULT_CONFIG


def send_traffic(api_url: str, n: int, drift: bool, config_path: str) -> None:
    """Replay held-out rows against the API, optionally shifted to simulate a changed population."""
    import httpx
    import pandas as pd

    from mlworkflow.config import load_config
    from mlworkflow.data import split

    cfg = load_config(config_path)
    test = split(pd.read_parquet(cfg.raw_path), cfg).X_test.sample(n, random_state=0)
    if drift:
        # An older, longer-working population: shifts two features the model relies on.
        test["age"] = (test["age"] + 15).clip(upper=90)
        test["hours-per-week"] = (test["hours-per-week"] * 1.3).clip(upper=99).round()
    records = json.loads(test.to_json(orient="records"))
    with httpx.Client(base_url=api_url, timeout=30) as client:
        for start in range(0, len(records), 100):
            r = client.post("/predict", json={"records": records[start:start + 100]})
            r.raise_for_status()
    print(f"sent {len(records)} records (drift={drift}) to {api_url}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="mlwf")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("train", help="run the train-and-promote flow once")

    p = sub.add_parser("drift", help="compare logged requests with the training data")
    p.add_argument("--last-n", type=int, default=None)

    p = sub.add_parser("traffic", help="send held-out rows to the API")
    p.add_argument("--api-url", default="http://localhost:8000")
    p.add_argument("-n", type=int, default=500)
    p.add_argument("--drift", action="store_true")

    p = sub.add_parser("schedule", help="serve both flows on cron schedules (blocks)")
    p.add_argument("--train-cron", default="0 3 * * 1")   # Mondays 03:00
    p.add_argument("--drift-cron", default="0 * * * *")   # hourly

    args = parser.parse_args(argv)
    if args.cmd == "train":
        from mlworkflow.pipeline import train_flow
        print(json.dumps(train_flow(args.config), indent=2))
    elif args.cmd == "drift":
        from mlworkflow.pipeline import drift_flow
        report = drift_flow(args.config, args.last_n)
        print(json.dumps({k: v for k, v in report.items() if k != "features"}, indent=2))
    elif args.cmd == "traffic":
        send_traffic(args.api_url, args.n, args.drift, args.config)
    elif args.cmd == "schedule":
        from prefect import serve

        from mlworkflow.pipeline import drift_flow, train_flow
        serve(
            train_flow.to_deployment("weekly-retrain", cron=args.train_cron,
                                     parameters={"config_path": args.config}),
            drift_flow.to_deployment("hourly-drift", cron=args.drift_cron,
                                     parameters={"config_path": args.config}),
        )


if __name__ == "__main__":
    main()
