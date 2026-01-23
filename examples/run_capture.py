#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OrcaGen 采集示例入口：python examples/run_capture.py 或 python -m examples.run_capture
"""
from __future__ import annotations

import os
import sys


def _ensure_project_root() -> None:
    script_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(script_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def main() -> None:
    _ensure_project_root()
    from orcagen.core.config import build_arg_parser, CaptureConfig
    from examples.capture_runner import CaptureRunner

    parser = build_arg_parser()
    args = parser.parse_args()
    config = CaptureConfig.from_args(args)
    runner = CaptureRunner(config)
    runner.run()


if __name__ == "__main__":
    main()

