#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OrcaGen 采集示例入口：python examples/run_capture.py 或 python -m examples.run_capture
"""
from __future__ import annotations

import argparse
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
    from examples.capture_runner import CaptureRunner, MultiGroupCaptureRunner
    from orcagen.utils.ids import now_id, default_object_id

    parser = build_arg_parser()
    args = parser.parse_args()
    if args.object_sites_groups:
        groups = [g.strip() for g in args.object_sites_groups.split("|") if g.strip()]
        group_configs = []
        base_sequence_id = None
        if args.sequence_id:
            base_sequence_id = args.sequence_id
        elif getattr(args, "sequence_prefix", None):
            base_sequence_id = f"{args.sequence_prefix}_{now_id()}"
        elif groups:
            first_name = groups[0].split(",")[0].strip()
            base_sequence_id = f"sequence_capture_{default_object_id(first_name)}_{now_id()}"
        for idx, g in enumerate(groups):
            local_args = argparse.Namespace(**vars(args))
            local_args.object_sites_groups = None
            local_args.object_sites = g
            local_args.object_site = None
            local_args.video_subdir = "rgb_main"
            if args.sequence_id:
                local_args.sequence_id = f"{args.sequence_id}_g{idx}"
            elif base_sequence_id:
                local_args.sequence_id = f"{base_sequence_id}_g{idx}"
            config = CaptureConfig.from_args(local_args)
            group_configs.append(config)
        runner = MultiGroupCaptureRunner(group_configs)
        runner.run()
    else:
        config = CaptureConfig.from_args(args)
        runner = CaptureRunner(config)
        runner.run()


if __name__ == "__main__":
    main()

