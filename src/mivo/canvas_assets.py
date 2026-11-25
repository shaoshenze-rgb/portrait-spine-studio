"""Bounded image handoff from real Canvas results; no model or network calls."""
from __future__ import annotations

import base64
import binascii
import hashlib
import io
from typing import Sequence

import numpy as np
from PIL import Image

from src.studio.vision_schema import VisionValidationError

MAX_RGBA_BYTES = 24 * 1024 * 1024
MAX_RGBA_PIXELS = 64_000_000


def decode_canvas_completion(value: dict, *, image_size: Sequence[int], input_sha256: str) -> np.ndarray:
    if not isinstance(value,dict):raise VisionValidationError('provided需要Canvas材质交接对象。')
    keys={'png_base64','sha256','input_sha256','canvas_scene_id','canvas_node_id','canvas_task_id',
          'operation','source_asset_sha256','result_asset_sha256'}
    if set(value)!=keys:raise VisionValidationError('Canvas材质交接字段缺失或未知。')
    if value['input_sha256']!=input_sha256:raise VisionValidationError('Canvas材质未绑定当前输入。')
    for key in ('sha256','source_asset_sha256','result_asset_sha256'):
        v=value[key]
        if not isinstance(v,str) or len(v)!=64 or any(c not in '0123456789abcdef' for c in v):
            raise VisionValidationError('Canvas材质摘要非法：'+key)
    for key in ('canvas_scene_id','canvas_node_id','canvas_task_id'):
        if not isinstance(value[key],str) or not value[key].strip() or len(value[key])>256:
            raise VisionValidationError('Canvas材质缺少真实交接标识：'+key)
    if value['operation'] not in ('split-layers','edit-image'):
        raise VisionValidationError('Canvas材质必须来自拆层或补全改图回执。')
    encoded=value['png_base64']
    if not isinstance(encoded,str) or len(encoded)>4*((MAX_RGBA_BYTES+2)//3):
        raise VisionValidationError('Canvas材质编码超限。')
    try:raw=base64.b64decode(encoded,validate=True)
    except (binascii.Error,ValueError) as exc:raise VisionValidationError('Canvas材质base64非法。') from exc
    if len(raw)>MAX_RGBA_BYTES or hashlib.sha256(raw).hexdigest()!=value['sha256']:
        raise VisionValidationError('Canvas材质字节摘要不匹配。')
    if not raw.startswith(bytes([137,80,78,71,13,10,26,10])):
        raise VisionValidationError('Canvas交接材质必须是RGBA PNG。')
    if len(image_size)!=2 or image_size[0]*image_size[1]>MAX_RGBA_PIXELS:
        raise VisionValidationError('Canvas交接画布尺寸超限。')
    try:
        with Image.open(io.BytesIO(raw)) as image:
            if image.format!='PNG' or image.mode!='RGBA' or image.size!=tuple(image_size):
                raise VisionValidationError('Canvas材质未对齐当前画布（禁止隐式缩放/去底）。')
            rgba=np.asarray(image,dtype=np.uint8).copy()
    except OSError as exc:raise VisionValidationError('Canvas材质PNG无法解码。') from exc
    if not (rgba[...,3]>0).any():raise VisionValidationError('Canvas材质为空。')
    return rgba


def completion_provenance(value: dict) -> dict:
    return {k:v for k,v in value.items() if k!='png_base64'}
