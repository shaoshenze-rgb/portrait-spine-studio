"""Image-bound Cindy vision bridge client; production never falls back to geometry."""
from __future__ import annotations

import base64
import hashlib
import json
import re
import stat
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

from PIL import Image


class VisionProviderError(RuntimeError):
    def __init__(self, message: str, *, code: str = 'VISION_BRIDGE_UNAVAILABLE',
                 retryable: bool = True, details: Optional[dict] = None):
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.details = details or {}


@dataclass
class VisionAnalysis:
    source: str
    result: Optional[Dict[str, Any]]
    metadata: Dict[str, Any]


class GeometryVisionProvider:
    """Explicit injection for offline tests/diagnosis, never chosen by HTTP payloads."""
    source = 'geometry'
    diagnostic_mode = True

    def analyze(self, *, job_id: str, image_path: Path, out_dir: Path,
                preset: str, on_status: Optional[Callable[[str], None]] = None) -> VisionAnalysis:
        return VisionAnalysis('geometry', None, {
            'provider': 'explicit_geometry_injection',
            'input_sha256': hashlib.sha256(image_path.read_bytes()).hexdigest(),
            'automatically_recognized': False,
        })


class ConversationHandoffProvider:
    """Current Cindy conversation supplies analysis; this app does not call a model."""
    source = 'conversation_handoff'

    def __init__(self, *, required_contract_version: int = 2):
        if required_contract_version not in (1, 2) or isinstance(required_contract_version, bool):
            raise ValueError('Unsupported analysis contract version')
        self.required_contract_version = required_contract_version
        # V1 remains available only through explicit constructor injection for old diagnostics.
        self.diagnostic_mode = required_contract_version == 1

    def prepare(self, *, job_id: str, image_path: Path, out_dir: Path, preset: str) -> dict:
        with Image.open(image_path) as image:
            size = list(image.size)
        request = {
            'version': self.required_contract_version, 'job_id': job_id,
            'input_sha256': hashlib.sha256(image_path.read_bytes()).hexdigest(),
            'image_size': size, 'image_path': str(image_path.resolve()),
            'image_url': f'/jobs/{job_id}/matted.png', 'preset': preset,
            'expected_result_path': str((out_dir / 'vision.json').resolve()),
            'source': self.source, 'application_invoked_model': False,
            'required_contract_version': self.required_contract_version,
            'contract_kind': 'complete_semantic_modules' if not self.diagnostic_mode else 'legacy_diagnostic',
            'diagnostic_mode': self.diagnostic_mode,
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        CindyVisionProvider._write(out_dir / 'analysis-request.json', request)
        return request

    def apply(self, *, job_id: str, image_path: Path, out_dir: Path,
              request: dict, payload: dict) -> VisionAnalysis:
        from .vision_schema import VisionValidationError, validate_vision_result
        result = payload.get('analysis')
        if (payload.get('job_id') != job_id or not isinstance(result, dict)
                or result.get('job_id') != job_id or request.get('job_id') != job_id):
            raise VisionValidationError('分析结果的 job_id 与当前任务不匹配。')
        if (isinstance(result.get('version'), bool)
                or result.get('version') != self.required_contract_version):
            raise VisionProviderError(
                '当前流程需要完整语义模块 V2：人体、武器和软配件都要明确归属；旧 V1 关键点或主体大掩码不能自动补成完整模块。',
                code='MODULE_CONTRACT_REQUIRED', retryable=False,
                details={'required_version': self.required_contract_version, 'received_version': result.get('version')},
            )
        with Image.open(image_path) as image:
            size = list(image.size)
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        if request.get('input_sha256') != digest or request.get('image_size') != size:
            raise VisionValidationError('待分析图片已变化，不能把旧分析应用到当前输入。')
        out_dir.mkdir(parents=True, exist_ok=True)
        CindyVisionProvider._write(out_dir / 'submitted-analysis.json', payload)
        normalized = validate_vision_result(result, input_sha256=digest, image_size=size)
        CindyVisionProvider._write(out_dir / 'vision.json', normalized)
        return VisionAnalysis(self.source, normalized, {
            'source': self.source, 'application_invoked_model': False,
            'automatically_recognized': False,
            'analysis_origin': 'current_cindy_conversation_handoff',
            'contract_version': normalized['version'], 'diagnostic_mode': self.diagnostic_mode,
            'job_id': job_id, 'input_sha256': digest, 'image_size': size,
            'result_path': str((out_dir / 'vision.json').resolve()),
        })


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class CindyVisionProvider:
    source = 'actual_cindy_agent'
    MAX_RESPONSE_BYTES = 4 * 1024 * 1024

    def __init__(self, connection_path: Union[str, Path], *, timeout_s: float = 900,
                 request_timeout_s: float = 15, poll_interval_s: float = 1):
        self.connection_path = Path(connection_path).resolve()
        self.timeout_s = timeout_s
        self.request_timeout_s = request_timeout_s
        self.poll_interval_s = poll_interval_s
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())

    def _connection(self) -> dict:
        try:
            if stat.S_IMODE(self.connection_path.stat().st_mode) & 0o077:
                raise VisionProviderError('视觉连接文件权限过宽，请重新启动 Cindy 连接插件。', code='VISION_CONNECTION_INSECURE', retryable=False)
            cfg = json.loads(self.connection_path.read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            raise VisionProviderError('尚未连接本机 Cindy 视觉插件。请启动连接插件后重试当前识别阶段。') from exc
        if not isinstance(cfg, dict):
            raise VisionProviderError('本机视觉连接配置无效。', code='VISION_CONNECTION_INVALID', retryable=False)
        try:
            url = urllib.parse.urlsplit(cfg.get('baseUrl', ''))
            valid = (url.scheme == 'http' and url.hostname == '127.0.0.1' and url.port
                     and not url.username and not url.password and url.path in ('', '/')
                     and not url.query and not url.fragment
                     and cfg.get('schemaVersion') == 1 and cfg.get('protocol') == 'portrait-vision-v1'
                     and isinstance(cfg.get('instanceId'), str) and bool(cfg['instanceId'])
                     and isinstance(cfg.get('token'), str) and len(cfg['token']) >= 24
                     and '\n' not in cfg['token'] and '\r' not in cfg['token'])
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise VisionProviderError('视觉连接仅允许有效的本机 127.0.0.1 插件端点。', code='VISION_CONNECTION_INVALID', retryable=False)
        return cfg

    def _request(self, cfg: dict, method: str, path: str, body: Optional[dict] = None) -> dict:
        raw = None if body is None else json.dumps(body, ensure_ascii=False).encode('utf-8')
        req = urllib.request.Request(cfg['baseUrl'].rstrip('/') + path, data=raw, method=method,
                                     headers={'Authorization': 'Bearer ' + cfg['token'], 'Content-Type': 'application/json'})
        try:
            with self._opener.open(req, timeout=self.request_timeout_s) as response:
                data = response.read(self.MAX_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as exc:
            try:
                failure = json.loads(exc.read(self.MAX_RESPONSE_BYTES))
                error = failure.get('error') or {}
            except (ValueError, AttributeError):
                error = {}
            code = str(error.get('code') or 'VISION_BRIDGE_HTTP_ERROR')
            # Never include request headers, a configured secret, or arbitrary server dumps.
            message = str(error.get('message') or '本机视觉插件拒绝了请求。')[:1000].replace(cfg['token'], '[已隐藏]')
            raise VisionProviderError(message, code=code, retryable=bool(error.get('retryable', False)), details={'http_status': exc.code}) from exc
        except (OSError, urllib.error.URLError) as exc:
            raise VisionProviderError('暂时无法取得本机视觉插件回执；保留原任务编号，重试只查询该任务。', code='VISION_BRIDGE_TRANSPORT', details={'outcome_unknown': True}) from exc
        if len(data) > self.MAX_RESPONSE_BYTES:
            raise VisionProviderError('视觉插件返回数据超过上限，已停止处理。', code='VISION_RESPONSE_TOO_LARGE', retryable=False)
        try:
            result = json.loads(data)
        except (ValueError, UnicodeError) as exc:
            raise VisionProviderError('视觉插件返回的不是有效 JSON。', code='VISION_RESPONSE_INVALID', retryable=False) from exc
        if not isinstance(result, dict):
            raise VisionProviderError('视觉插件返回结构无效。', code='VISION_RESPONSE_INVALID', retryable=False)
        return result

    @staticmethod
    def _write(path: Path, value: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    @staticmethod
    def _check_binding(response: dict, binding: dict) -> None:
        if any(response.get(k) != v for k, v in binding.items()):
            raise VisionProviderError('视觉回执与当前任务、输入图片或尺寸不一致。', code='VISION_BINDING_MISMATCH', retryable=False)
        if response.get('source') != 'actual_cindy_agent':
            raise VisionProviderError('视觉回执没有真实 Cindy Agent 来源。', code='VISION_SOURCE_INVALID', retryable=False)

    def analyze(self, *, job_id: str, image_path: Path, out_dir: Path,
                preset: str, on_status: Optional[Callable[[str], None]] = None) -> VisionAnalysis:
        from .vision_schema import VisionValidationError, validate_vision_result
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', job_id):
            raise VisionProviderError('视觉任务编号无效。', code='VISION_JOB_INVALID', retryable=False)
        image_path, out_dir = Path(image_path), Path(out_dir)
        image_bytes = image_path.read_bytes()
        if len(image_bytes) > 10 * 1024 * 1024:
            raise VisionProviderError('视觉输入 PNG 超过连接插件的 10 MiB 上限。', code='VISION_INPUT_TOO_LARGE', retryable=False)
        digest = hashlib.sha256(image_bytes).hexdigest()
        with Image.open(image_path) as image:
            width, height = image.size
            if image.format != 'PNG':
                raise VisionProviderError('视觉输入必须是预处理后的 PNG。', code='VISION_INPUT_INVALID', retryable=False)
        binding = {'jobId': job_id, 'inputSha256': digest, 'width': width, 'height': height}
        cfg = self._connection()
        out_dir.mkdir(parents=True, exist_ok=True)
        state_path = out_dir / 'request-state.json'
        state = json.loads(state_path.read_text()) if state_path.is_file() else None
        if state and (state.get('binding') != binding or state.get('instanceId') != cfg['instanceId']):
            raise VisionProviderError('此前视觉任务的输入或插件实例发生变化，无法确认旧任务结果；已停止重复派发。', code='VISION_OUTCOME_UNKNOWN', retryable=False)
        path = '/v1/jobs/' + urllib.parse.quote(job_id, safe='')
        if state is None:
            # Write before dispatch: a lost response must not create a second paid/model task.
            state = {'binding': binding, 'instanceId': cfg['instanceId'], 'submitted': True}
            self._write(state_path, state)
            if on_status:
                on_status('正在提交本机 Cindy 视觉识别')
            try:
                response = self._request(cfg, 'POST', '/v1/jobs', {**binding, 'imageBase64': base64.b64encode(image_bytes).decode('ascii'), 'preset': preset})
            except VisionProviderError as exc:
                if exc.code == 'VISION_BRIDGE_TRANSPORT':
                    # One read-only reconciliation; a missing/unknown job is not permission to resend.
                    response = self._request(cfg, 'GET', path)
                else:
                    raise
        else:
            response = self._request(cfg, 'GET', path)
        deadline = time.monotonic() + self.timeout_s
        while True:
            self._check_binding(response, binding)
            self._write(out_dir / 'response.json', response)
            status = response.get('status')
            if status == 'completed':
                receipt = response.get('executionReceipt')
                if (not isinstance(receipt, dict) or any(not receipt.get(k) for k in ('taskId', 'runId', 'outputMessageId', 'acceptedConfig', 'execution'))):
                    raise VisionProviderError('视觉任务缺少真实执行回执，不能作为自动识别结果。', code='VISION_RECEIPT_MISSING', retryable=False)
                execution = receipt['execution']
                if (any(not isinstance(receipt[k], str) for k in ('taskId', 'runId', 'outputMessageId'))
                        or not isinstance(receipt['acceptedConfig'], dict)
                        or not isinstance(execution, dict) or not isinstance(execution.get('instanceId'), str)
                        or not execution['instanceId'] or isinstance(execution.get('generation'), bool)
                        or not isinstance(execution.get('generation'), int)):
                    raise VisionProviderError('视觉执行回执缺少有效配置或执行实例身份。', code='VISION_RECEIPT_MISSING', retryable=False)
                if response.get('taskId') != receipt['taskId'] or response.get('runId') != receipt['runId']:
                    raise VisionProviderError('视觉任务与执行回执身份不一致。', code='VISION_RECEIPT_MISMATCH', retryable=False)
                try:
                    result = validate_vision_result(response.get('result'), input_sha256=digest, image_size=[width, height])
                except VisionValidationError as exc:
                    exc.details = {'source': 'actual_cindy_agent', 'task_id': receipt['taskId'],
                                   'run_id': receipt['runId'], 'output_message_id': receipt['outputMessageId'],
                                   'input_sha256': digest, 'response_path': str(out_dir / 'response.json')}
                    raise
                self._write(out_dir / 'validated-result.json', result)
                return VisionAnalysis('actual_cindy_agent', result, {
                    'input_sha256': digest, 'image_size': [width, height], 'job_id': job_id,
                    'task_id': receipt['taskId'], 'run_id': receipt['runId'],
                    'output_message_id': receipt['outputMessageId'], 'execution_receipt': receipt,
                    'automatically_recognized': True, 'result_path': str(out_dir / 'validated-result.json'),
                })
            if status in ('failed', 'outcome_unknown'):
                error = response.get('error') or {}
                raise VisionProviderError(str(error.get('message') or '视觉识别失败或结果未确认，已保留任务与输入。')[:1000].replace(cfg['token'], '[已隐藏]'),
                                          code=str(error.get('code') or 'VISION_OUTCOME_UNKNOWN'),
                                          retryable=False, details={'bridge_status': status, 'task_id': response.get('taskId'), 'run_id': response.get('runId')})
            if status not in ('queued', 'running'):
                raise VisionProviderError('视觉插件返回未知状态。', code='VISION_RESPONSE_INVALID', retryable=False)
            if on_status:
                on_status('等待 Cindy 视觉任务执行' if status == 'queued' else 'Cindy 正在识别人形、宠物与软附属物')
            if time.monotonic() >= deadline:
                raise VisionProviderError('视觉任务等待超时，结果尚未确认。重试将查询原任务，不会重新提交。', code='VISION_OUTCOME_UNKNOWN', retryable=True)
            time.sleep(self.poll_interval_s)
            response = self._request(cfg, 'GET', path)
