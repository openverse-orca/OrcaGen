from __future__ import annotations

from typing import List, Optional, Tuple


def find_body_name(model, preferred: Optional[str], keywords: List[str]) -> str:
    body_names = list(model.get_body_names())
    if preferred:
        if preferred in body_names:
            return preferred
        for b in body_names:
            if b.lower() == preferred.lower():
                return b
    lower = [b.lower() for b in body_names]
    for kw in keywords:
        kwl = kw.lower()
        for b, bl in zip(body_names, lower):
            if kwl in bl:
                return b
    raise RuntimeError(
        f"找不到 body。preferred={preferred!r}, keywords={keywords!r}. "
        f"可用 body 数量={len(body_names)}，可先打印 model.get_body_names() 排查。"
    )


def ensure_site_dict(env) -> dict:
    try:
        site_dict = env.model.get_site_dict()
        if site_dict:
            return site_dict
    except Exception:
        pass
    try:
        site_dict = env.gym.query_all_sites()
        env.model.init_site_dict(site_dict)
        return site_dict
    except Exception:
        pass
    raise RuntimeError("无法获取 site_dict（既不在 env.model，也无法 query_all_sites）。")


def find_site_name(env, preferred: Optional[str], keywords: List[str]) -> str:
    site_dict = ensure_site_dict(env)
    site_names = list(site_dict.keys())
    if preferred:
        if preferred in site_names:
            return preferred
        for s in site_names:
            if s.lower() == preferred.lower():
                return s
    lower = [s.lower() for s in site_names]
    for kw in keywords:
        kwl = kw.lower()
        for s, sl in zip(site_names, lower):
            if kwl in sl:
                return s
    raise RuntimeError(
        f"找不到 site。preferred={preferred!r}, keywords={keywords!r}. "
        f"可用 site 数量={len(site_names)}，可先打印 site 名称排查。"
    )


def ensure_geom_dict(env) -> dict:
    try:
        geom_dict = env.model.get_geom_dict()
        if geom_dict:
            return geom_dict
    except Exception:
        pass
    if hasattr(env, "query_all_geoms"):
        geom_dict = env.query_all_geoms()
        try:
            env.model.init_geom_dict(geom_dict)
        except Exception:
            pass
        return geom_dict
    if hasattr(env.gym, "query_all_geoms"):
        geom_dict = env.gym.query_all_geoms()
        try:
            env.model.init_geom_dict(geom_dict)
        except Exception:
            pass
        return geom_dict
    raise RuntimeError("无法获取 geom_dict（既不在 env.model，也无法 query_all_geoms）。")


def infer_bbox_size_for_body(env, body_name: str, fallback_xyz: Tuple[float, float, float]) -> Tuple[float, float, float]:
    geom_dict = ensure_geom_dict(env)
    for _, g in geom_dict.items():
        if g.get("BodyName") != body_name:
            continue
        size = g.get("Size")
        if size is None:
            continue
        try:
            s = [float(x) for x in list(size)]
        except Exception:
            continue
        if len(s) >= 3 and (s[1] > 0 or s[2] > 0):
            return (2.0 * s[0], 2.0 * s[1], 2.0 * s[2])
        if len(s) >= 2 and s[1] > 0:
            r = s[0]
            half = s[1]
            return (2.0 * r, 2.0 * r, 2.0 * (half + r))
        if len(s) >= 1 and s[0] > 0:
            r = s[0]
            return (2.0 * r, 2.0 * r, 2.0 * r)
    return fallback_xyz


def infer_bbox_size_for_site(env, site_name: str, fallback_xyz: Tuple[float, float, float]) -> Tuple[float, float, float]:
    try:
        size_dict = env.query_site_size([site_name])
        size = size_dict.get(site_name, None)
        if size is not None:
            s = [float(x) for x in list(size)]
            if len(s) >= 3:
                return (2.0 * s[0], 2.0 * s[1], 2.0 * s[2])
            if len(s) >= 1:
                return (2.0 * s[0], 2.0 * s[0], 2.0 * s[0])
    except Exception:
        pass
    return fallback_xyz

