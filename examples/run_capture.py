#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OrcaGen 采集示例入口：python examples/run_capture.py 或 python -m examples.run_capture
"""
from __future__ import annotations

import argparse
import os
import sys
import socket
from typing import Optional


def _ensure_project_root() -> None:
    script_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(script_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def _extract_prefix_from_groups(groups_str: str) -> Optional[str]:
    """从 object_sites_groups 中提取前缀（例如：'Bin1|Bin2' -> 'bin'）"""
    if not groups_str:
        return None
    groups = [g.strip() for g in groups_str.split("|") if g.strip()]
    if not groups:
        return None
    
    # 提取第一个组的第一个物体名称
    first_group = groups[0]
    first_item = first_group.split(",")[0].strip()
    
    # 尝试提取前缀（去除数字后缀）
    # 例如：Bin1 -> bin, CardBoxA_01 -> cardboxa
    import re
    # 去除数字后缀
    prefix = re.sub(r'_\d+$', '', first_item, flags=re.IGNORECASE)
    prefix = re.sub(r'\d+$', '', prefix, flags=re.IGNORECASE)
    # 转换为小写并清理
    prefix = prefix.lower().strip('_')
    return prefix if prefix else None


def _interactive_prompt(args: argparse.Namespace) -> argparse.Namespace:
    """交互式提示用户输入关键参数"""
    print("\n[OrcaGen] ========== 交互式参数配置 ==========")
    
    # 1. 录制时间
    if args.duration_s == 30.0:  # 使用默认值时才提示
        try:
            user_input = input(f"录制时长（秒，默认 {args.duration_s}）: ").strip()
            if user_input:
                args.duration_s = float(user_input)
                print(f"  已设置 duration_s = {args.duration_s}")
        except (ValueError, KeyboardInterrupt):
            print(f"  使用默认值: {args.duration_s}")
    
    # 2. external_drive（带连接检查）
    if not args.external_drive:  # 未指定时才提示
        try:
            # 先检查连接
            is_connected = _check_grpc_connection(args.orcagym_addr)
            if is_connected:
                print(f"  ✓ 已检测到 gRPC 连接: {args.orcagym_addr}")
            else:
                print(f"  ⚠ 未检测到 gRPC 连接: {args.orcagym_addr}")
            
            user_input = input("仿真是否已由外部程序启动？(y/n，默认 n): ").strip().lower()
            if user_input in ('y', 'yes', '是'):
                args.external_drive = True
                print("  ✓ 已启用 external_drive 模式")
                if not is_connected:
                    print("  ⚠ 警告：未检测到连接，请确保仿真已启动")
        except KeyboardInterrupt:
            print("  使用默认值: external_drive=False")
    
    # 3. 自动提取 sequence_prefix
    if args.object_sites_groups and not args.sequence_prefix:
        prefix = _extract_prefix_from_groups(args.object_sites_groups)
        if prefix:
            try:
                user_input = input(f"检测到前缀 '{prefix}'，是否用作 sequence_prefix？(y/n，默认 y): ").strip().lower()
                if user_input in ('', 'y', 'yes', '是'):
                    args.sequence_prefix = prefix
                    print(f"  ✓ 已设置 sequence_prefix = '{prefix}'")
            except KeyboardInterrupt:
                pass
    
    print("=" * 50 + "\n")
    return args


def _check_grpc_connection(addr: str) -> bool:
    """检查 gRPC 连接是否可用"""
    try:
        host, port_str = addr.split(":")
        port = int(port_str)
        s = socket.socket()
        s.settimeout(1.0)
        s.connect((host, port))
        s.close()
        return True
    except Exception:
        return False


def main() -> None:
    _ensure_project_root()
    from orcagen.core.config import build_arg_parser, CaptureConfig
    from examples.capture_runner import CaptureRunner, MultiGroupCaptureRunner
    from orcagen.utils.ids import now_id, default_object_id

    parser = build_arg_parser()
    args = parser.parse_args()
    
    # 交互式输入（如果未通过命令行指定）
    # 可以通过环境变量 ORCAGEN_NO_INTERACTIVE=1 禁用交互式输入
    if os.getenv("ORCAGEN_NO_INTERACTIVE") != "1":
        args = _interactive_prompt(args)
    
    # 自动提取 sequence_prefix（如果未指定且提供了 object_sites_groups）
    if args.object_sites_groups and not args.sequence_prefix:
        prefix = _extract_prefix_from_groups(args.object_sites_groups)
        if prefix:
            args.sequence_prefix = prefix
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
            # 组索引从1开始（g1, g2, g3, g4）
            group_idx = idx + 1
            local_args.video_subdir = f"rgb_main_g{group_idx}"
            if args.sequence_id:
                local_args.sequence_id = f"{args.sequence_id}_g{group_idx}"
            elif base_sequence_id:
                local_args.sequence_id = f"{base_sequence_id}_g{group_idx}"
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

