"""Anti-ringing premultiplied Spine 4.2 rasterizer and dark/light MP4 preview writer."""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

from .spine_runtime import Skeleton


def find_ffmpeg(search_start: Optional[Union[str, Path]] = None) -> Optional[str]:
    """Auto-detect ffmpeg binary from PATH or workspace `tools/ffmpeg-local/`."""
    found = shutil.which("ffmpeg")
    if found:
        return found

    env_ff = os.environ.get("FFMPEG_PATH")
    if env_ff and Path(env_ff).is_file():
        return str(Path(env_ff))

    roots: List[Path] = []
    for base in [Path(search_start).resolve() if search_start else Path.cwd().resolve(), Path(__file__).resolve()]:
        cur = base if base.is_dir() else base.parent
        for _ in range(10):
            if cur not in roots:
                roots.append(cur)
            if cur.parent == cur:
                break
            cur = cur.parent

    for r in roots:
        ff_dir = r / "tools" / "ffmpeg-local"
        if not ff_dir.is_dir():
            continue
        txt = ff_dir / "ffmpeg-path.txt"
        if txt.is_file():
            p = txt.read_text(encoding="utf-8").strip()
            if p and Path(p).is_file():
                return p
        bins = ff_dir / "package" / "imageio_ffmpeg" / "binaries"
        if bins.is_dir():
            for cand in sorted(bins.glob("ffmpeg*")):
                if cand.is_file():
                    return str(cand)
    return None


def kernel_weights(f: np.ndarray, kind: str) -> Tuple[List[int], np.ndarray]:
    if kind == "cubic":  # Catmull-Rom (Keys a=-0.5), interpolating
        f2 = f * f
        f3 = f2 * f
        w = np.stack(
            [
                -0.5 * f3 + f2 - 0.5 * f,
                1.5 * f3 - 2.5 * f2 + 1.0,
                -1.5 * f3 + 2.0 * f2 + 0.5 * f,
                0.5 * f3 - 0.5 * f2,
            ],
            axis=1,
        )
        return [-1, 0, 1, 2], w
    if kind == "lanczos3":
        offs = np.arange(-2, 4)
        x = f[:, None] - offs[None, :]
        w = np.where(np.abs(x) < 1e-7, 1.0, np.sinc(x) * np.sinc(x / 3.0))
        w = w / w.sum(axis=1, keepdims=True)
        return list(offs), w
    raise ValueError(f"Unsupported sampling kernel: {kind!r}")


def sample_texture(tex: np.ndarray, tx: np.ndarray, ty: np.ndarray, kind: str = "cubic") -> np.ndarray:
    """Sample premultiplied RGBA float32 texture with anti-ringing 2x2 neighborhood clamp."""
    h, w = tex.shape[:2]
    ix = np.floor(tx).astype(np.int64)
    iy = np.floor(ty).astype(np.int64)
    fx = (tx - ix).astype(np.float32)
    fy = (ty - iy).astype(np.float32)
    ox, wx = kernel_weights(fx, kind)
    oy, wy = kernel_weights(fy, kind)
    flat = tex.reshape(-1, 4)
    out = np.zeros((len(tx), 4), dtype=np.float32)
    cols = [np.clip(ix + o, 0, w - 1) for o in ox]
    for j, o in enumerate(oy):
        row = np.clip(iy + o, 0, h - 1) * w
        acc = np.zeros((len(tx), 4), dtype=np.float32)
        for i in range(len(ox)):
            acc += flat[row + cols[i]] * wx[:, i : i + 1]
        out += acc * wy[:, j : j + 1]
    x0, x1 = np.clip(ix, 0, w - 1), np.clip(ix + 1, 0, w - 1)
    y0, y1 = np.clip(iy, 0, h - 1) * w, np.clip(iy + 1, 0, h - 1) * w
    q0 = flat[y0 + x0]
    q1 = flat[y0 + x1]
    q2 = flat[y1 + x0]
    q3 = flat[y1 + x1]
    lo = np.minimum(np.minimum(q0, q1), np.minimum(q2, q3))
    hi = np.maximum(np.maximum(q0, q1), np.maximum(q2, q3))
    return np.clip(out, lo, hi)


def raster_uv(
    vc: np.ndarray,
    uv: np.ndarray,
    tris: np.ndarray,
    width: int,
    height: int,
    ss: int = 1,
) -> Tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
    """Rasterize textured triangles in canvas coordinates onto the ss-supersampled grid."""
    P = vc * ss
    x0 = max(0, int(math.floor(float(P[:, 0].min()))) - 1)
    x1 = min(width * ss, int(math.ceil(float(P[:, 0].max()))) + 1)
    y0 = max(0, int(math.floor(float(P[:, 1].min()))) - 1)
    y1 = min(height * ss, int(math.ceil(float(P[:, 1].max()))) + 1)
    if x1 <= x0 or y1 <= y0:
        return 0, 0, np.zeros((0, 0), dtype=bool), np.zeros((0, 0)), np.zeros((0, 0))
    hh, ww = y1 - y0, x1 - x0
    U = np.zeros((hh, ww), dtype=np.float64)
    V = np.zeros((hh, ww), dtype=np.float64)
    M = np.zeros((hh, ww), dtype=bool)
    eps = -1e-9
    for a, b, c in tris:
        pa, pb, pc = P[a], P[b], P[c]
        tx0 = max(x0, int(math.floor(min(pa[0], pb[0], pc[0]) - 0.5)))
        tx1 = min(x1, int(math.ceil(max(pa[0], pb[0], pc[0]) - 0.5)) + 1)
        ty0 = max(y0, int(math.floor(min(pa[1], pb[1], pc[1]) - 0.5)))
        ty1 = min(y1, int(math.ceil(max(pa[1], pb[1], pc[1]) - 0.5)) + 1)
        if tx1 <= tx0 or ty1 <= ty0:
            continue
        xs = np.arange(tx0, tx1) + 0.5
        ys = np.arange(ty0, ty1) + 0.5
        v0 = pb - pa
        v1 = pc - pa
        d = v0[0] * v1[1] - v1[0] * v0[1]
        if abs(d) < 1e-12:
            continue
        X = xs[None, :] - pa[0]
        Y = ys[:, None] - pa[1]
        l1 = (X * v1[1] - v1[0] * Y) / d
        l2 = (v0[0] * Y - X * v0[1]) / d
        l0 = 1.0 - l1 - l2
        inside = (l0 >= eps) & (l1 >= eps) & (l2 >= eps)
        if not inside.any():
            continue
        u = l0 * uv[a, 0] + l1 * uv[b, 0] + l2 * uv[c, 0]
        v = l0 * uv[a, 1] + l1 * uv[b, 1] + l2 * uv[c, 1]
        sl = (slice(ty0 - y0, ty1 - y0), slice(tx0 - x0, tx1 - x0))
        U[sl] = np.where(inside, u, U[sl])
        V[sl] = np.where(inside, v, V[sl])
        M[sl] |= inside
    return y0, x0, M, U, V


def over_bg(frame_pm: np.ndarray, bg: Tuple[int, int, int] = (17, 17, 17)) -> np.ndarray:
    """Composite premultiplied uint8 RGBA frame over solid RGB background."""
    a = frame_pm[..., 3:4].astype(np.float32) / 255.0
    rgb = frame_pm[..., :3].astype(np.float32) + np.array(bg, dtype=np.float32) * (1.0 - a)
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def straight_alpha(frame_pm: np.ndarray) -> np.ndarray:
    """Un-premultiply uint8 RGBA frame into straight-alpha uint8 RGBA."""
    out = np.zeros_like(frame_pm, dtype=np.uint8)
    a = frame_pm[..., 3]
    out[..., 3] = a
    nz = a > 0
    if nz.any():
        rgb = frame_pm[..., :3][nz].astype(np.float32) * 255.0 / a[nz, None].astype(np.float32)
        out[..., :3][nz] = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    return out


def parts_composite_pm(parts_dir: Union[str, Path], width: int, height: int) -> np.ndarray:
    """Premultiplied source-over composite of raw parts from parts.json at their bboxes."""
    pdir = Path(parts_dir)
    pj = json.loads((pdir / "parts.json").read_text(encoding="utf-8"))
    acc = np.zeros((height, width, 4), dtype=np.float32)
    for p in sorted(pj["parts"], key=lambda x: x["drawOrder"]):
        a = np.asarray(Image.open(pdir / p["image"]).convert("RGBA"), dtype=np.float32)
        pm = a.copy()
        pm[..., :3] *= a[..., 3:4] / 255.0
        x, y, w, h = p["bbox"]
        sub = acc[y : y + h, x : x + w]
        acc[y : y + h, x : x + w] = pm + sub * (1.0 - pm[..., 3:4] / 255.0)
    out = np.clip(np.rint(acc), 0, 255).astype(np.uint8)
    out[..., :3] = np.minimum(out[..., :3], out[..., 3:4])
    return out


def _resolve_spine_json(rig_dir: Path, spine_json_path: Optional[Union[str, Path]] = None) -> Path:
    if spine_json_path is not None:
        return Path(spine_json_path)
    rep_path = rig_dir / "rig-build-report.json"
    if rep_path.is_file():
        rep = json.loads(rep_path.read_text(encoding="utf-8"))
        sf = rep.get("spineFile")
        if sf and (rig_dir / sf).is_file():
            return rig_dir / sf
    cands = sorted(rig_dir.glob("*.spine.json"))
    if not cands:
        raise FileNotFoundError(f"No *.spine.json found in {rig_dir}")
    return cands[0]


class SpineRenderer:
    """Offline Spine 4.2 rasterizer returning premultiplied RGBA uint8 frames."""

    def __init__(
        self,
        rig_dir: Union[str, Path],
        spine_json_path: Optional[Union[str, Path]] = None,
        kernel: str = "cubic",
        ss: int = 1,
        anim: str = "idle",
        render_scale: float = 1.0,
    ):
        self.rig_dir = Path(rig_dir).resolve()
        self.spine_path = _resolve_spine_json(self.rig_dir, spine_json_path)
        self.kernel = kernel
        self.ss = ss
        self.anim = anim
        self.sk = Skeleton(self.spine_path)

        sk_meta = self.sk.json["skeleton"]
        self.width = int(round(float(sk_meta["width"])))
        self.height = int(round(float(sk_meta["height"])))
        self.origin_x = -float(sk_meta.get("x", -self.width / 2.0))
        self.origin_y = float(self.height) + float(sk_meta.get("y", 0.0))
        self.fps = int(sk_meta.get("fps", 30))

        rep_path = self.rig_dir / "rig-build-report.json"
        if rep_path.is_file():
            rep = json.loads(rep_path.read_text(encoding="utf-8"))
            if "origin" in rep:
                self.origin_x, self.origin_y = float(rep["origin"][0]), float(rep["origin"][1])
            if "canvas" in rep:
                self.width, self.height = int(rep["canvas"][0]), int(rep["canvas"][1])

        self.source_width = self.width
        self.source_height = self.height
        scale = float(render_scale)
        if not (0.0 < scale <= 1.0):
            raise ValueError(f"render_scale must be in (0, 1], got {render_scale}")
        self.render_scale = scale
        if scale < 1.0:
            self.width = max(2, int(round(self.source_width * scale)) // 2 * 2)
            self.height = max(2, int(round(self.source_height * scale)) // 2 * 2)
        self._canvas_sx = self.width / float(self.source_width)
        self._canvas_sy = self.height / float(self.source_height)

        img_rel = sk_meta.get("images", "./images/")
        self.tex: Dict[str, np.ndarray] = {}
        for slot in self.sk.slots:
            att_name = slot["attachment"]
            p = self.rig_dir / img_rel / f"{att_name}.png"
            a = np.asarray(Image.open(p).convert("RGBA"), dtype=np.float32)
            pm = a.copy()
            pm[..., :3] *= a[..., 3:4] / 255.0
            self.tex[slot["name"]] = pm

    def to_canvas(self, world: np.ndarray) -> np.ndarray:
        cx = world[:, 0] + self.origin_x
        cy = self.origin_y - world[:, 1]
        if self.render_scale < 1.0:
            return np.stack([cx * self._canvas_sx, cy * self._canvas_sy], axis=1)
        return np.stack([cx, cy], axis=1)

    def pose(self, t: Optional[float] = None, gain: float = 1.0) -> None:
        if t is None:
            self.sk.setup_pose()
            self.sk.update_world()
        else:
            self.sk.apply(self.anim, t, gain=gain)

    def frame(self, t: Optional[float] = None, gain: float = 1.0, *, slot_names=None) -> np.ndarray:
        """Render at time t (None = setup pose). Returns uint8 (H, W, 4) premultiplied."""
        self.pose(t, gain=gain)
        ss = self.ss
        acc = np.zeros((self.height * ss, self.width * ss, 4), dtype=np.float32)
        for slot in self.sk.slots:
            if slot_names is not None and slot['name'] not in slot_names:
                continue
            world, uv, tris, _att = self.sk.attachment_vertices(slot)
            vc = self.to_canvas(world)
            y0, x0, M, U, V = raster_uv(vc, uv, tris, self.width, self.height, ss)
            if not M.any():
                continue
            tex = self.tex[slot["name"]]
            th, tw = tex.shape[:2]
            ys, xs = np.nonzero(M)
            col = sample_texture(tex, U[ys, xs] * tw - 0.5, V[ys, xs] * th - 0.5, self.kernel)
            col[:, 3] = np.clip(col[:, 3], 0.0, 255.0)
            col[:, :3] = np.clip(col[:, :3], 0.0, col[:, 3:4])
            gy, gx = ys + y0, xs + x0
            dst = acc[gy, gx]
            acc[gy, gx] = col + dst * (1.0 - col[:, 3:4] / 255.0)
        if ss > 1:
            acc = acc.reshape(self.height, ss, self.width, ss, 4).mean(axis=(1, 3))
        out = np.clip(np.rint(acc), 0, 255).astype(np.uint8)
        out[..., :3] = np.minimum(out[..., :3], out[..., 3:4])
        return out


def write_preview_mp4(
    frames_pm: Sequence[np.ndarray],
    out_path: Union[str, Path],
    bg: Tuple[int, int, int] = (17, 17, 17),
    fps: int = 30,
    loops: int = 1,
    ffmpeg_bin: Optional[str] = None,
) -> Path:
    """Encode premultiplied RGBA frames over solid `bg` into an H.264 MP4 preview."""
    ff = ffmpeg_bin or find_ffmpeg()
    if not ff:
        raise RuntimeError("ffmpeg binary not found in PATH or workspace tools/ffmpeg-local/")
    if not frames_pm:
        raise ValueError("frames_pm must not be empty")
    dst = Path(out_path).resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)

    h, w = frames_pm[0].shape[:2]
    # H.264 yuv420p requires even dimensions; pad 1px if odd.
    ew = w + (w % 2)
    eh = h + (h % 2)

    cmd = [
        ff,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{ew}x{eh}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-vf",
        "format=yuv420p",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-movflags",
        "+faststart",
        str(dst),
    ]
    with subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
        assert proc.stdin is not None
        for _ in range(max(1, loops)):
            for f in frames_pm:
                rgb = over_bg(f, bg)
                if ew != w or eh != h:
                    padded = np.full((eh, ew, 3), bg, dtype=np.uint8)
                    padded[:h, :w] = rgb
                    rgb = padded
                proc.stdin.write(rgb.tobytes())
        proc.stdin.close()
        stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg preview encode failed (rc={rc}): {stderr}")
    return dst
