"""Owner-operated local CLI for RSI status, configurations, approval, promotion and rollback."""

import argparse
import json
from collections.abc import Sequence

from .config import load_settings
from .rsi_store import RSIStore


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evorag-rsi")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    sub.add_parser("configurations")
    sub.add_parser("recommendations")

    args = parser.parse_args(argv)
    settings = load_settings()
    store = RSIStore(settings.storage.database_path, settings.rsi.storage)
    if args.command == "status":
        active = store.active_version()
        print(
            json.dumps(
                {
                    "enabled": settings.rsi.enabled,
                    "automatic_promotion": settings.rsi.automatic_promotion,
                    "active_version": active.version_id,
                    "total_cycles": store.count_cycles(),
                },
                indent=2,
            )
        )
        return 0
    if args.command == "configurations":
        print(json.dumps([v.model_dump(mode="json") for v in store.list_versions()], indent=2))
        return 0
    print(json.dumps([r.model_dump(mode="json") for r in store.list_recommendations()], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
