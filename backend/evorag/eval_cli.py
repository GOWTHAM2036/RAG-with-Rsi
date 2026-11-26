"""Lightweight CLI for dataset inspection, validation and offline evaluation reports."""

import argparse
import json
from collections.abc import Sequence

from .config import load_settings
from .dataset_registry import DatasetRegistry, validate_dataset
from .experiment_store import ExperimentStore


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evorag-eval")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list-datasets")
    sub.add_parser("validate-datasets")
    show = sub.add_parser("show-run")
    show.add_argument("run_id")

    args = parser.parse_args(argv)
    registry = DatasetRegistry()
    if args.command == "list-datasets":
        print(json.dumps([d.model_dump(mode="json") for d in registry.list_datasets()], indent=2))
        return 0
    if args.command == "validate-datasets":
        issues = {
            ds.dataset_id: validate_dataset(registry.get(ds.dataset_id))
            for ds in registry.list_datasets()
        }
        print(json.dumps(issues, indent=2))
        return 1 if any(issues.values()) else 0
    settings = load_settings()
    store = ExperimentStore(settings.storage.database_path, settings.evaluation.storage)
    res = store.get_run(args.run_id)
    if res is None:
        print(json.dumps({"error": "evaluation_run_not_found"}))
        return 1
    print(res.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
