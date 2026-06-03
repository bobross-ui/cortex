import sys
import json
import dataclasses
from pathlib import Path
from cortex.ingestion import resolve


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print("usage: python -m cortex.pipeline.ingest <export_dir>", file=sys.stderr)
        return 2
    root = Path(argv[0])
    parser = resolve(root)
    items = list(parser.parse(root))
    print(parser.report.human_summary())
    print(json.dumps(dataclasses.asdict(parser.report), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
