"""Command-line interface.

  python main.py evaluate [--verbose] [--output results.csv]
  python main.py decide --event event.json
"""
from __future__ import annotations

import argparse
import json
import sys

from logistics_agent.config import ConfigError, Settings
from logistics_agent.data_loader import build_cases, load_test_cases
from logistics_agent.evaluation import EvaluationRunner, ReasoningJudge
from logistics_agent.evaluation.reporting import console_step_printer
from logistics_agent.logging_setup import configure_logging
from logistics_agent.service import DecisionService


def _bootstrap():
    settings = Settings.from_env()
    configure_logging(settings.log_level)
    return settings, DecisionService(settings)


def cmd_evaluate(args: argparse.Namespace) -> int:
    settings, service = _bootstrap()
    cases = build_cases(load_test_cases(settings.test_cases_file))
    runner = EvaluationRunner(
        service,
        ReasoningJudge(service.llms.validator),
        step_observer=console_step_printer if args.verbose else None,
    )
    results = runner.run_all(cases)
    runner.print_summary(results)
    if args.output:
        results.to_csv(args.output, index=False)
    return 0


def cmd_decide(args: argparse.Namespace) -> int:
    _, service = _bootstrap()
    with open(args.event, "r", encoding="utf-8") as fh:
        event = json.load(fh)
    result = service.process_event(event)
    print(result.model_dump_json(indent=2))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Logistics disruption decision system")
    sub = parser.add_subparsers(dest="command", required=True)

    p_eval = sub.add_parser("evaluate", help="Run the curated test cases")
    p_eval.add_argument("--verbose", action="store_true", help="Print each workflow step")
    p_eval.add_argument("--output", help="Write the results table to a CSV file")
    p_eval.set_defaults(func=cmd_evaluate)

    p_dec = sub.add_parser("decide", help="Decide on a single event from a JSON file")
    p_dec.add_argument("--event", required=True, help="Path to a JSON file with the event")
    p_dec.set_defaults(func=cmd_decide)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
