"""Save the full regression result and the exact validated source manifest."""
from __future__ import annotations

import json
import time
import unittest
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    directory = root / "reports/parallel_20261003"
    started = time.perf_counter()
    suite = unittest.defaultTestLoader.discover(str(root / "tests"), pattern="test_*.py")
    with (directory / "tests.log").open("w", encoding="utf-8") as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    payload = {"passed": result.wasSuccessful(), "exit_code": 0 if result.wasSuccessful() else 1,
        "tests": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
        "skipped": len(result.skipped), "seconds": time.perf_counter() - started}
    (directory / "tests.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if result.wasSuccessful():
        from junqi_open_v4.pipeline import source_contract
        source = source_contract(root)
        (directory / "validated_source.json").write_text(json.dumps(source, indent=2), encoding="utf-8")
        for name in ("games", "arena", "pipeline", "teachers"):
            (directory / ("after_" + name + ".py")).write_bytes((root / "python/junqi_open_v4" / (name + ".py")).read_bytes())
    print(json.dumps(payload), flush=True)
    raise SystemExit(payload["exit_code"])


if __name__ == "__main__":
    main()
