#!/usr/bin/env python3
"""Folder-based 20-30 portrait batch orchestrator (4K-ready, hard-lock safe, resumable)."""
from __future__ import annotations

import argparse
import base64
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.export.export_profile import DEFAULT_EXPORT_PROFILE, FORMAT_OPTIONS, SIZE_PRESETS
from src.pipeline.local_matte import matte_portrait_file
from src.studio.batch_manager import BatchManager


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _make_requester(base_url: str, timeout: float = 600.0):
    url = urllib.parse.urlsplit(base_url)
    if (
        url.scheme != "http"
        or url.hostname not in ("127.0.0.1", "localhost")
        or url.username
        or url.password
    ):
        raise ValueError("--base-url 只能指向本机 Studio HTTP 服务")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    root = base_url.rstrip("/")

    def request(path: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            root + path,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with opener.open(req, timeout=timeout) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            try:
                err = json.loads(raw)
                msg = err.get("message") or raw
            except ValueError:
                msg = raw
            raise RuntimeError(f"Studio HTTP {exc.code}: {msg}") from exc

    return request


def _cmd_init(args: argparse.Namespace) -> int:
    work_dir = Path(args.work_dir).expanduser().resolve()
    bm = BatchManager(work_dir)

    profile_patch: Dict[str, Any] = {}
    if args.size_preset:
        profile_patch["size_preset"] = args.size_preset
    if args.width:
        profile_patch["width"] = args.width
    if args.height:
        profile_patch["height"] = args.height
    if args.max_export_edge:
        profile_patch["max_long_edge"] = args.max_export_edge
    if args.align:
        profile_patch["align"] = args.align
    if args.fps:
        profile_patch["fps"] = args.fps
    if args.formats:
        profile_patch["formats"] = [f.strip() for f in args.formats.split(",") if f.strip()]
    if args.output_dir:
        profile_patch["output_dir"] = str(Path(args.output_dir).expanduser().resolve())

    batch = bm.create_or_rescan_batch(
        args.input_dir,
        name=args.name,
        batch_id=args.batch,
        export_profile=profile_patch or None,
    )
    batch_id = batch["batch_id"]
    matted_dir = work_dir / "batches" / batch_id / "matted"
    matted_dir.mkdir(parents=True, exist_ok=True)

    # 1. Parallel local matting for unlocked items without matted PNG
    to_matte = [
        it for it in batch["items"]
        if not it.get("locked")
        and (not it.get("matted_path") or not Path(str(it["matted_path"])).is_file())
    ]

    def _run_matte(item: Dict[str, Any]) -> Dict[str, Any]:
        src = Path(item["source_path"])
        out_png = matted_dir / f"{item['character']}.png"
        chk_jpg = matted_dir / f"{item['character']}-check.jpg"
        info = matte_portrait_file(src, out_png, check_out=chk_jpg)
        return {
            "item_id": item["item_id"],
            "matted_path": str(out_png),
            "matte_check_path": str(chk_jpg),
            "matte_info": info,
        }

    matte_errors: List[Dict[str, str]] = []
    if to_matte:
        workers = max(1, min(int(args.workers), len(to_matte)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_run_matte, it): it for it in to_matte}
            for fut in as_completed(futs):
                it = futs[fut]
                try:
                    res = fut.result()
                    next_status = it["status"] if it["status"] not in ("pending", "failed") else "matted"
                    bm.update_item(
                        batch_id,
                        it["item_id"],
                        {
                            "matted_path": res["matted_path"],
                            "matte_check_path": res["matte_check_path"],
                            "status": next_status,
                            "error": None,
                        },
                    )
                except Exception as exc:
                    matte_errors.append({"item_id": it["item_id"], "error": str(exc)})
                    bm.update_item(
                        batch_id,
                        it["item_id"],
                        {"status": "failed", "error": f"去底失败：{exc}"},
                    )

    overview_path = bm.build_matte_overview(batch_id)

    # 2. Prepare Studio jobs (default 4096px max_long_edge to preserve 4K clarity)
    prepare_errors: List[Dict[str, str]] = []
    if not args.skip_prepare:
        req = _make_requester(args.base_url, timeout=args.timeout)
        req("/api/health")
        latest = bm.get_batch(batch_id)
        for it in latest["items"]:
            if it.get("locked") or it.get("job_id") or it.get("status") == "failed":
                continue
            mp = it.get("matted_path")
            if not mp or not Path(mp).is_file():
                continue
            try:
                png_bytes = Path(mp).read_bytes()
                job = req(
                    "/api/jobs",
                    {
                        "filename": Path(mp).name,
                        "image_base64": base64.b64encode(png_bytes).decode("ascii"),
                        "preset": args.preset,
                        "character": it["character"],
                        "pause_before_animation": True,
                        "max_long_edge": int(args.max_long_edge),
                        "wait_if_busy": True,
                        "batch_id": batch_id,
                    },
                )
                bm.update_item(
                    batch_id,
                    it["item_id"],
                    {"job_id": job["job_id"], "status": "prepared", "error": None},
                )
            except Exception as exc:
                prepare_errors.append({"item_id": it["item_id"], "error": str(exc)})
                bm.update_item(
                    batch_id,
                    it["item_id"],
                    {"status": "failed", "error": f"建任务失败：{exc}"},
                )

    final_batch = bm.get_batch(batch_id)
    pending_split = [
        {
            "item_id": it["item_id"],
            "character": it["character"],
            "job_id": it.get("job_id"),
            "matted_path": it.get("matted_path"),
            "status": it.get("status"),
        }
        for it in final_batch["items"]
        if not it.get("locked") and it.get("status") in ("matted", "prepared", "needs_fix")
    ]
    out = {
        "ok": True,
        "batch_id": batch_id,
        "name": final_batch["name"],
        "source_dir": final_batch["source_dir"],
        "batch_url": f"{args.base_url.rstrip('/')}/?batch={batch_id}",
        "matte_overview_path": str(overview_path) if overview_path else None,
        "export_profile": final_batch["export_profile"],
        "counts": bm.compute_counts(final_batch),
        "rescan_summary": batch.get("rescan_summary"),
        "pending_split_count": len(pending_split),
        "pending_split_items": pending_split,
        "matte_errors": matte_errors,
        "prepare_errors": prepare_errors,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    try:
        req = _make_requester(args.base_url, timeout=args.timeout)
        view = req(f"/api/batches/{urllib.parse.quote(args.batch)}")
        view["batch_url"] = f"{args.base_url.rstrip('/')}/?batch={args.batch}"
        print(json.dumps(view, ensure_ascii=False, indent=2))
        return 0
    except Exception:
        bm = BatchManager(Path(args.work_dir).expanduser().resolve())
        batch = bm.get_batch(args.batch)
        print(
            json.dumps(
                {
                    "ok": True,
                    "offline": True,
                    "batch_id": batch["batch_id"],
                    "counts": bm.compute_counts(batch),
                    "items": batch["items"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0


def _cmd_fix_list(args: argparse.Namespace) -> int:
    bm = BatchManager(Path(args.work_dir).expanduser().resolve())
    items = bm.get_fix_list(args.batch)
    print(
        json.dumps(
            {"ok": True, "batch_id": args.batch, "count": len(items), "items": items},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _cmd_sync(args: argparse.Namespace) -> int:
    bm = BatchManager(Path(args.work_dir).expanduser().resolve())
    req = _make_requester(args.base_url, timeout=args.timeout)
    batch = bm.get_batch(args.batch)
    updated = 0
    for it in batch["items"]:
        if it.get("locked") or not it.get("job_id"):
            continue
        job = req(f"/api/jobs/{urllib.parse.quote(it['job_id'])}")
        st = job.get("status")
        if st in ("canvas_preview", "ready") and it.get("status") != "needs_fix":
            bm.update_item(args.batch, it["item_id"], {"status": "preview_ready", "error": None})
            updated += 1
        elif st == "failed":
            bm.update_item(
                args.batch,
                it["item_id"],
                {"status": "failed", "error": (job.get("stage_error") or {}).get("message") or "任务失败"},
            )
            updated += 1
    final_batch = bm.get_batch(args.batch)
    print(
        json.dumps(
            {
                "ok": True,
                "batch_id": args.batch,
                "updated": updated,
                "counts": bm.compute_counts(final_batch),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    req = _make_requester(args.base_url, timeout=args.timeout)
    payload: Dict[str, Any] = {"async": False}
    prof: Dict[str, Any] = {}
    if args.size_preset:
        prof["size_preset"] = args.size_preset
    if args.width:
        prof["width"] = args.width
    if args.height:
        prof["height"] = args.height
    if args.align:
        prof["align"] = args.align
    if args.fps:
        prof["fps"] = args.fps
    if args.formats:
        prof["formats"] = [f.strip() for f in args.formats.split(",") if f.strip()]
    if args.output_dir:
        out_abs = str(Path(args.output_dir).expanduser().resolve())
        prof["output_dir"] = out_abs
        payload["output_dir"] = out_abs
    if prof:
        payload["export_profile"] = prof
    if args.max_frames is not None:
        payload["max_frames"] = args.max_frames
    res = req(f"/api/batches/{urllib.parse.quote(args.batch)}/export", payload)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0 if res.get("error_count", 0) == 0 else 1


def _cmd_list(args: argparse.Namespace) -> int:
    bm = BatchManager(Path(args.work_dir).expanduser().resolve())
    print(json.dumps({"ok": True, "batches": bm.list_batches()}, ensure_ascii=False, indent=2))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Portrait Spine Studio 文件夹批量任务总管（支持 4K / 锁定防改 / 打回返修）")
    parser.add_argument("--work-dir", default=str(REPO_ROOT / ".studio-work"), help="Studio 工作目录")
    parser.add_argument("--base-url", default="http://127.0.0.1:8765", help="本机 Studio 地址")
    parser.add_argument("--timeout", type=float, default=900.0, help="HTTP 超时秒数")

    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="扫描文件夹、批量去底、生成总览图并创建 Studio 任务")
    p_init.add_argument("--input-dir", required=True, help="包含立绘图片的本机文件夹路径")
    p_init.add_argument("--name", help="批次显示名（默认使用文件夹名）")
    p_init.add_argument("--batch", help="已有批次 ID（传此项做幂等增量重扫，已锁定角色绝不修改）")
    p_init.add_argument("--max-long-edge", type=int, default=4096, help="预处理长边上限（默认 4096 保留 4K 清晰度）")
    p_init.add_argument("--preset", choices=["agile_full", "heavy_weapon_full", "bust_half"], default="agile_full")
    p_init.add_argument("--size-preset", choices=list(SIZE_PRESETS), default=DEFAULT_EXPORT_PROFILE["size_preset"])
    p_init.add_argument("--width", type=int)
    p_init.add_argument("--height", type=int)
    p_init.add_argument("--max-export-edge", type=int)
    p_init.add_argument("--align", choices=["bottom_center", "center"], default=DEFAULT_EXPORT_PROFILE["align"])
    p_init.add_argument("--fps", type=int, choices=[15, 24, 30, 60], default=DEFAULT_EXPORT_PROFILE["fps"])
    p_init.add_argument(
        "--formats",
        default=",".join(DEFAULT_EXPORT_PROFILE["formats"]),
        help=f"逗号分隔的默认导出格式，可选：{','.join(FORMAT_OPTIONS)}",
    )
    p_init.add_argument("--output-dir", help="默认批量导出输出目录")
    p_init.add_argument("--workers", type=int, default=3, help="并行去底线程数（默认 3）")
    p_init.add_argument("--skip-prepare", action="store_true", help="只做去底与总览图，暂不提交 Studio prepare")

    p_status = sub.add_parser("status", help="查看批次实时状态与统计")
    p_status.add_argument("--batch", required=True)

    p_fix = sub.add_parser("fix-list", help="仅列出被用户标记「打回待修」且未锁定的角色及备注")
    p_fix.add_argument("--batch", required=True)

    p_sync = sub.add_parser("sync", help="从 Studio 同步批内所有未锁定角色的最新预览状态")
    p_sync.add_argument("--batch", required=True)

    p_exp = sub.add_parser("export", help="按统一规格批量导出批内所有已确认锁定的角色")
    p_exp.add_argument("--batch", required=True)
    p_exp.add_argument("--output-dir", help="导出目录")
    p_exp.add_argument("--size-preset", choices=list(SIZE_PRESETS))
    p_exp.add_argument("--width", type=int)
    p_exp.add_argument("--height", type=int)
    p_exp.add_argument("--align", choices=["bottom_center", "center"])
    p_exp.add_argument("--fps", type=int, choices=[15, 24, 30, 60])
    p_exp.add_argument("--formats", help=f"逗号分隔格式：{','.join(FORMAT_OPTIONS)}")
    p_exp.add_argument("--max-frames", type=int, help="调试用：仅渲染前 N 帧")

    sub.add_parser("list", help="列出本机所有批次")

    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            return _cmd_init(args)
        if args.command == "status":
            return _cmd_status(args)
        if args.command == "fix-list":
            return _cmd_fix_list(args)
        if args.command == "sync":
            return _cmd_sync(args)
        if args.command == "export":
            return _cmd_export(args)
        if args.command == "list":
            return _cmd_list(args)
    except Exception as exc:
        print(json.dumps({"ok": False, "message": str(exc)}, ensure_ascii=False, indent=2))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
