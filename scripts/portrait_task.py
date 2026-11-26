#!/usr/bin/env python3
"""Current Cindy conversation -> local HTTP job; no model invocation or plugin install."""
from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from src.gates.module_quality import RULES_VERSION


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='通过当前 Cindy 对话准备立绘、提交分析并查看本机任务。应用不直接调用模型。')
    parser.add_argument('--base-url', default='http://127.0.0.1:8765', help='本机 Studio 地址；默认 http://127.0.0.1:8765')
    parser.add_argument('--timeout', type=float, default=600, help='HTTP等待秒数，不会自动重复提交')
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare', help='上传立绘，预处理后停在 awaiting_analysis')
    prepare.add_argument('--image', required=True, type=Path)
    prepare.add_argument('--character', help='Studio 任务和输出文件使用的角色标识')
    prepare.add_argument('--preset', choices=['agile_full', 'heavy_weapon_full', 'bust_half'], default='agile_full')
    apply = sub.add_parser('apply-analysis', help='应用当前对话读图产出的同任务视觉JSON，停在绑定预览')
    apply.add_argument('--job', required=True)
    apply.add_argument('--analysis', required=True, type=Path, help='完整模块 V2 JSON 文件，额外含 job_id')
    canvas = sub.add_parser('apply-canvas', help='现有Canvas图层直接生成预览')
    canvas.add_argument('--job', required=True)
    canvas.add_argument('--parts-dir', required=True, type=Path)
    canvas.add_argument('--plan', required=True, type=Path)
    canvas.add_argument('--cleanup', action='store_true')
    generate = sub.add_parser('generate', help='确认已有绑定后生成动画；仅在本次任务已授权验收时使用')
    generate.add_argument('--job', required=True)
    generate.add_argument('--module-review', required=True, type=Path, help='实际查看当前原图及每个完整模块后的观察JSON，绑定确认与语义复核一次提交')
    status = sub.add_parser('status', help='只读查询任务')
    status.add_argument('--job', required=True)
    review = sub.add_parser('review', help='为已实际观察的同版本动画记录视觉复核；机器待审或失败不可放行')
    review.add_argument('--job', required=True)
    review.add_argument('--decision', choices=['pass', 'fail', 'pending'], required=True)
    review.add_argument('--reviewer', required=True)
    review.add_argument('--notes-file', required=True, type=Path)
    review.add_argument('--binding-file', required=True, type=Path, help='观察前保存的 status JSON 或 review_binding JSON，禁止静默绑定新版本')
    review.add_argument('--observations-file', required=True, type=Path, help='五项视觉观察状态与具体依据的 JSON')
    review.add_argument('--full-loop-observed', action='store_true')
    review.add_argument('--original-and-modules-viewed', action='store_true')
    args = parser.parse_args(argv)
    url = urllib.parse.urlsplit(args.base_url)
    if (url.scheme != 'http' or url.hostname not in ('127.0.0.1', 'localhost')
            or url.username or url.password or url.path not in ('', '/') or url.query or url.fragment):
        parser.error('--base-url 只能是本机 Studio 的 HTTP 地址')
    if hasattr(args, 'job') and not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', args.job):
        parser.error('--job 无效')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def request(path, payload=None):
        data = None if payload is None else json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(args.base_url.rstrip('/') + path, data=data,
                                     headers={'Content-Type': 'application/json'})
        with opener.open(req, timeout=args.timeout) as response:
            return json.load(response)

    try:
        health = request('/api/health')
        if health.get('vision_provider') != 'conversation_handoff':
            raise ValueError('当前端口不是对话交接版 Studio，请先启动本项目最新工具页；未提交输入。')
        if health.get('analysis_contract_version') != 2 or health.get('legacy_diagnostic_mode') is True:
            raise ValueError('当前端口不支持完整模块 V2 生产合同；未提交输入或默退旧几何诊断。')
        if health.get('module_qc_rules_version') != RULES_VERSION:
            raise ValueError('当前服务仍加载旧QC规则，请启动已审核的当前版本后再继续；未提交请求。')
        if args.command == 'prepare':
            output = request('/api/jobs', {
                'filename': args.image.name,
                'image_base64': base64.b64encode(args.image.read_bytes()).decode('ascii'),
                'preset': args.preset, 'character': args.character, 'pause_before_animation': True,
            })
        elif args.command == 'apply-analysis':
            analysis = json.loads(args.analysis.read_text(encoding='utf-8'))
            output = request('/api/jobs/' + args.job + '/apply-analysis', {'job_id': args.job, 'analysis': analysis})
        elif args.command == 'apply-canvas':
            output = request('/api/jobs/' + args.job + '/apply-canvas-layers', {
                'parts_dir': str(args.parts_dir.resolve()), 'plan': str(args.plan.resolve()), 'cleanup': args.cleanup})
        elif args.command == 'generate':
            output = request('/api/jobs/' + args.job + '/confirm-rig',
                             {'module_review': json.loads(args.module_review.read_text(encoding='utf-8'))})
        elif args.command == 'review':
            saved = json.loads(args.binding_file.read_text(encoding='utf-8'))
            binding = saved.get('review_binding', saved)
            output = request('/api/jobs/' + args.job + '/review', {
                'decision': args.decision, 'source': 'current_cindy_conversation',
                'reviewer': args.reviewer, 'notes': args.notes_file.read_text(encoding='utf-8'),
                'full_loop_observed': args.full_loop_observed, 'review_binding': binding,
                'original_and_modules_viewed': args.original_and_modules_viewed,
                'observations': json.loads(args.observations_file.read_text(encoding='utf-8')),
            })
        else:
            output = request('/api/jobs/' + args.job)
        output['page_url'] = args.base_url.rstrip('/') + '/?job=' + output['job_id']
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 1 if output.get('status') == 'failed' else 0
    except urllib.error.HTTPError as exc:
        try:
            error = json.loads(exc.read())
        except (ValueError, UnicodeError):
            error = {'message': f'本机 Studio HTTP {exc.code}'}
        print(json.dumps({'ok': False, 'http_status': exc.code, 'error': error}, ensure_ascii=False, indent=2))
        return 1
    except (OSError, ValueError, urllib.error.URLError) as exc:
        print(json.dumps({'ok': False, 'message': str(exc), 'submitted_again': False}, ensure_ascii=False, indent=2))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
