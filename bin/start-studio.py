#!/usr/bin/env python3
"""One-command local launcher for Portrait Spine Studio."""
from __future__ import annotations

import argparse
import sys
import threading
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.studio.server import create_studio_server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="启动 Portrait Spine Studio 本机离线工具页")
    parser.add_argument("--host", default="127.0.0.1", help="监听主机（默认 127.0.0.1）")
    parser.add_argument("--port", type=int, default=8765, help="监听端口（默认 8765，传 0 自动分配）")
    parser.add_argument(
        "--work-dir",
        default=str(REPO_ROOT / ".studio-work"),
        help="任务输出与缓存工作目录（默认 <repo>/.studio-work）",
    )
    parser.add_argument("--no-open", action="store_true", help="启动后不自动打开浏览器")
    args = parser.parse_args(argv)

    server = create_studio_server(host=args.host, port=args.port, work_dir=args.work_dir)
    actual_port = server.server_address[1]
    url = f"http://{args.host}:{actual_port}/"

    print(f"[Portrait Spine Studio] 本地离线服务已启动：{url}", flush=True)
    print(f"[Portrait Spine Studio] 工作目录：{Path(args.work_dir).resolve()}", flush=True)
    print("[Portrait Spine Studio] 默认等待当前 Cindy 对话读图分析；应用不直接调用模型，无需安装插件。", flush=True)
    print("[Portrait Spine Studio] 按 Ctrl+C 停止服务。", flush=True)

    if not args.no_open:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Portrait Spine Studio] 服务已停止。", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
