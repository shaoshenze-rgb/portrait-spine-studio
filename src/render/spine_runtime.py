"""Clean Spine 4.2 JSON runtime (NumPy): FK, 1/2-bone IK, weighted meshes, exact cubic bezier."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


def _bezier_value(t: float, t0: float, v0: float, t1: float, v1: float, c: Sequence[float]) -> float:
    """Solve Spine 4 cubic Bezier curve [cx1, cy1, cx2, cy2] in absolute (time, value) units."""
    cx1, cy1, cx2, cy2 = c
    lo, hi = 0.0, 1.0
    for _ in range(60):
        s = 0.5 * (lo + hi)
        u = 1.0 - s
        x = u * u * u * t0 + 3.0 * u * u * s * cx1 + 3.0 * u * s * s * cx2 + s * s * s * t1
        if x < t:
            lo = s
        else:
            hi = s
    s = 0.5 * (lo + hi)
    u = 1.0 - s
    return u * u * u * v0 + 3.0 * u * u * s * cy1 + 3.0 * u * s * s * cy2 + s * s * s * v1


def _sample(keys: List[Dict[str, Any]], t: float, fields: Sequence[str], defaults: Sequence[float]) -> List[float]:
    times = [float(k.get("time", 0.0)) for k in keys]
    vals = [[float(k.get(f, d)) for f, d in zip(fields, defaults)] for k in keys]
    if t <= times[0]:
        return vals[0]
    if t >= times[-1]:
        return vals[-1]
    i = max(j for j in range(len(times)) if times[j] <= t)
    k = keys[i]
    t0, t1 = times[i], times[i + 1]
    curve = k.get("curve")
    if curve == "stepped":
        return vals[i]
    out: List[float] = []
    for n in range(len(fields)):
        v0, v1 = vals[i][n], vals[i + 1][n]
        if isinstance(curve, list):
            out.append(_bezier_value(t, t0, v0, t1, v1, curve[4 * n : 4 * n + 4]))
        else:
            out.append(v0 + (v1 - v0) * (t - t0) / (t1 - t0))
    return out


def _deform_at(keys: List[Dict[str, Any]], t: float) -> np.ndarray:
    times = [float(k.get("time", 0.0)) for k in keys]

    def arr(k: Dict[str, Any]) -> Tuple[int, List[float]]:
        return int(k.get("offset", 0)), list(k.get("vertices", []))

    if t <= times[0] or len(keys) == 1:
        i, s = 0, 0.0
    elif t >= times[-1]:
        i, s = len(keys) - 1, 0.0
    else:
        i = max(j for j in range(len(times)) if times[j] <= t)
        c = keys[i].get("curve")
        t0, t1 = times[i], times[i + 1]
        if c == "stepped":
            s = 0.0
        elif isinstance(c, list):
            s = _bezier_value(t, t0, 0.0, t1, 1.0, c[:4])
        else:
            s = (t - t0) / (t1 - t0)

    def dense(k: Dict[str, Any], n: int) -> np.ndarray:
        out = np.zeros(n, dtype=np.float64)
        off, vs = arr(k)
        out[off : off + len(vs)] = vs
        return out

    n = max(int(k.get("offset", 0)) + len(k.get("vertices", [])) for k in keys)
    a = dense(keys[i], n)
    if s and i + 1 < len(keys):
        a = a + (dense(keys[i + 1], n) - a) * s
    return a


class Bone:
    def __init__(self, data: Dict[str, Any], parent: Optional["Bone"]):
        self.name: str = data["name"]
        self.parent: Optional["Bone"] = parent
        self.data: Dict[str, Any] = data
        self.x: float = 0.0
        self.y: float = 0.0
        self.rotation: float = 0.0
        self.scaleX: float = 1.0
        self.scaleY: float = 1.0
        self.shearX: float = 0.0
        self.shearY: float = 0.0
        self.a: float = 1.0
        self.b: float = 0.0
        self.c: float = 0.0
        self.d: float = 1.0
        self.worldX: float = 0.0
        self.worldY: float = 0.0
        self.setup()

    def setup(self) -> None:
        d = self.data
        self.x = float(d.get("x", 0.0))
        self.y = float(d.get("y", 0.0))
        self.rotation = float(d.get("rotation", 0.0))
        self.scaleX = float(d.get("scaleX", 1.0))
        self.scaleY = float(d.get("scaleY", 1.0))
        self.shearX = float(d.get("shearX", 0.0))
        self.shearY = float(d.get("shearY", 0.0))

    def update_world(self, root_xy: Tuple[float, float] = (0.0, 0.0)) -> None:
        rx = math.radians(self.rotation + self.shearX)
        ry = math.radians(self.rotation + 90.0 + self.shearY)
        la = math.cos(rx) * self.scaleX
        lb = math.cos(ry) * self.scaleY
        lc = math.sin(rx) * self.scaleX
        ld = math.sin(ry) * self.scaleY
        p = self.parent
        if p is None:
            self.a, self.b, self.c, self.d = la, lb, lc, ld
            self.worldX = self.x + root_xy[0]
            self.worldY = self.y + root_xy[1]
        else:
            inherit = self.data.get("inherit", self.data.get("transform", "normal"))
            if inherit != "normal":
                raise ValueError(f"Unsupported bone inherit mode {inherit!r} on bone {self.name}")
            self.a = p.a * la + p.b * lc
            self.b = p.a * lb + p.b * ld
            self.c = p.c * la + p.d * lc
            self.d = p.c * lb + p.d * ld
            self.worldX = p.a * self.x + p.b * self.y + p.worldX
            self.worldY = p.c * self.x + p.d * self.y + p.worldY

    def matrix(self) -> np.ndarray:
        return np.array([[self.a, self.b, self.worldX], [self.c, self.d, self.worldY]], dtype=np.float64)

    def tip_world(self) -> np.ndarray:
        length = float(self.data.get("length", 0.0))
        return np.array([self.a * length + self.worldX, self.c * length + self.worldY], dtype=np.float64)


class Skeleton:
    def __init__(self, path_or_dict: Union[str, Path, Dict[str, Any]]):
        if isinstance(path_or_dict, dict):
            d = path_or_dict
        else:
            d = json.loads(Path(path_or_dict).read_text(encoding="utf-8"))
        self.json: Dict[str, Any] = d
        self.bones: List[Bone] = []
        self.by_name: Dict[str, Bone] = {}
        for bd in d["bones"]:
            b = Bone(bd, self.by_name.get(bd.get("parent")))
            self.bones.append(b)
            self.by_name[b.name] = b
        self.slots: List[Dict[str, Any]] = d["slots"]
        skin = next(s for s in d["skins"] if s["name"] == "default")
        self.attachments: Dict[str, Dict[str, Any]] = skin["attachments"]
        self.bone_index: Dict[str, int] = {b.name: i for i, b in enumerate(self.bones)}
        self.ik: List[Dict[str, Any]] = sorted(d.get("ik", []), key=lambda c: c.get("order", 0))
        self.children: Dict[str, List[Bone]] = {b.name: [] for b in self.bones}
        for b in self.bones:
            if b.parent is not None:
                self.children[b.parent.name].append(b)
        self.deform: Dict[Tuple[str, str], np.ndarray] = {}

    def setup_pose(self) -> None:
        for b in self.bones:
            b.setup()
        self.deform = {}

    def apply(self, anim_name: str, t: float, gain: float = 1.0) -> None:
        """Apply setup pose + animation at time t (optionally scaling offsets by gain for stress tests)."""
        self.setup_pose()
        anim = self.json["animations"][anim_name]
        for bname, tls in anim.get("bones", {}).items():
            b = self.by_name[bname]
            sd = b.data
            for kind, keys in tls.items():
                if kind == "rotate":
                    b.rotation = float(sd.get("rotation", 0.0)) + gain * _sample(keys, t, ["value"], [0.0])[0]
                elif kind == "translate":
                    x, y = _sample(keys, t, ["x", "y"], [0.0, 0.0])
                    b.x = float(sd.get("x", 0.0)) + gain * x
                    b.y = float(sd.get("y", 0.0)) + gain * y
                elif kind == "translatex":
                    b.x = float(sd.get("x", 0.0)) + gain * _sample(keys, t, ["value"], [0.0])[0]
                elif kind == "translatey":
                    b.y = float(sd.get("y", 0.0)) + gain * _sample(keys, t, ["value"], [0.0])[0]
                elif kind == "scale":
                    x, y = _sample(keys, t, ["x", "y"], [1.0, 1.0])
                    b.scaleX = float(sd.get("scaleX", 1.0)) * (1.0 + gain * (x - 1.0))
                    b.scaleY = float(sd.get("scaleY", 1.0)) * (1.0 + gain * (y - 1.0))
                elif kind == "shear":
                    x, y = _sample(keys, t, ["x", "y"], [0.0, 0.0])
                    b.shearX = float(sd.get("shearX", 0.0)) + gain * x
                    b.shearY = float(sd.get("shearY", 0.0)) + gain * y
                else:
                    raise ValueError(f"Unsupported bone timeline {kind!r}")
        for _skin_name, slots in anim.get("attachments", {}).items():
            for slot, atts in slots.items():
                for att, tls in atts.items():
                    keys = tls.get("deform")
                    if keys:
                        self.deform[(slot, att)] = gain * _deform_at(keys, t)
        self.update_world()

    def update_world(self) -> None:
        for b in self.bones:
            b.update_world()
        for c in self.ik:
            self._apply_ik(c)

    def _update_subtree(self, bone: Bone) -> None:
        bone.update_world()
        for ch in self.children[bone.name]:
            self._update_subtree(ch)

    def _apply_ik(self, c: Dict[str, Any]) -> None:
        names = c["bones"]
        tgt = self.by_name[c["target"]]
        T = np.array([tgt.worldX, tgt.worldY], dtype=np.float64)
        ang = lambda v: math.degrees(math.atan2(v[1], v[0]))
        if c.get("mix", 1.0) != 1.0 or c.get("softness", 0) or c.get("stretch") or c.get("compress"):
            raise ValueError(f"Unsupported IK options in {c['name']}")
        if len(names) not in (1, 2):
            raise ValueError(f"IK requires one or two bones in {c['name']}")

        def parent_space(b: Bone, point: np.ndarray) -> np.ndarray:
            if b.parent is None:
                return point
            m = b.parent.matrix()
            try:
                return np.linalg.solve(m[:, :2], point - m[:, 2])
            except np.linalg.LinAlgError as exc:
                raise ValueError(f"Singular IK ancestor transform in {c['name']}") from exc

        if len(names) == 1:
            b = self.by_name[names[0]]
            if abs(b.scaleX) < 1e-12:
                raise ValueError(f"Zero IK bone scale in {c['name']}")
            delta = parent_space(b, T) - np.array([b.x, b.y])
            if np.linalg.norm(delta) > 1e-12:
                b.rotation = ang(delta) - b.shearX - (180.0 if b.scaleX < 0 else 0.0)
            self._update_subtree(b)
            return

        p, ch = self.by_name[names[0]], self.by_name[names[1]]
        if ch.parent is not p:
            raise ValueError(f"IK bones must form a parent-child chain in {c['name']}")
        for b in (p, ch):
            if (
                b.scaleX <= 0
                or not math.isclose(b.scaleX, b.scaleY, rel_tol=1e-9, abs_tol=1e-12)
                or abs(b.shearX) > 1e-9
                or abs(b.shearY) > 1e-9
            ):
                raise ValueError(f"Unsupported IK chain scale/shear on {b.name} in {c['name']}")
        delta = parent_space(p, T) - np.array([p.x, p.y])
        l1 = math.hypot(ch.x, ch.y) * p.scaleX
        l2 = float(ch.data.get("length", 0.0)) * p.scaleX * ch.scaleX
        if l1 < 1e-12 or l2 < 1e-12:
            raise ValueError(f"IK chain lengths must be positive in {c['name']}")
        d = float(np.linalg.norm(delta))
        cosa = (l1 * l1 + d * d - l2 * l2) / (2.0 * l1 * max(d, 1e-9))
        alpha = math.degrees(math.acos(min(1.0, max(-1.0, cosa))))
        bend = 1.0 if c.get("bendPositive", True) else -1.0
        offset = ang([ch.x, ch.y])
        direction = ang(delta) if d > 1e-12 else p.rotation + offset
        p.rotation = direction - bend * alpha - offset
        p.update_world()
        ch.update_world()
        child_delta = parent_space(ch, T) - np.array([ch.x, ch.y])
        if np.linalg.norm(child_delta) > 1e-12:
            ch.rotation = ang(child_delta)
        self._update_subtree(p)

    def attachment_vertices(self, slot: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """World (Spine y-up) vertex positions (N,2), uvs (N,2), triangles (M,3), attachment dict."""
        name = slot.get("attachment")
        att = self.attachments.get(slot["name"], {}).get(name)
        if att is None:
            raise KeyError(f"Missing attachment {name!r} for slot {slot['name']!r}")
        bone = self.by_name[slot["bone"]]
        typ = att.get("type", "region")
        if typ == "region":
            w, h = float(att["width"]), float(att["height"])
            sx, sy = float(att.get("scaleX", 1.0)), float(att.get("scaleY", 1.0))
            r = math.radians(float(att.get("rotation", 0.0)))
            c, s = math.cos(r), math.sin(r)
            loc = np.array([[-w / 2, -h / 2], [-w / 2, h / 2], [w / 2, h / 2], [w / 2, -h / 2]], dtype=np.float64)
            loc *= np.array([sx, sy], dtype=np.float64)
            loc = loc @ np.array([[c, s], [-s, c]], dtype=np.float64) + np.array(
                [float(att.get("x", 0.0)), float(att.get("y", 0.0))], dtype=np.float64
            )
            M = bone.matrix()
            world = loc @ M[:, :2].T + M[:, 2]
            uvs = np.array([[0, 1], [0, 0], [1, 0], [1, 1]], dtype=np.float64)
            tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
            return world, uvs, tris, att
        if typ == "mesh":
            uvs = np.asarray(att["uvs"], dtype=np.float64).reshape(-1, 2)
            n = len(uvs)
            v = att["vertices"]
            dfm = self.deform.get((slot["name"], name))
            if len(v) == 2 * n:
                loc = np.asarray(v, dtype=np.float64).reshape(-1, 2)
                if dfm is not None:
                    loc = loc + np.asarray(dfm, dtype=np.float64).reshape(-1, 2)
                M = bone.matrix()
                world = loc @ M[:, :2].T + M[:, 2]
            else:
                world = np.zeros((n, 2), dtype=np.float64)
                i = 0
                f = 0
                mats = [b.matrix() for b in self.bones]
                for k in range(n):
                    cnt = int(v[i])
                    i += 1
                    acc = np.zeros(2, dtype=np.float64)
                    for _ in range(cnt):
                        bi = int(v[i])
                        bx, by, wt = float(v[i + 1]), float(v[i + 2]), float(v[i + 3])
                        i += 4
                        if dfm is not None:
                            bx += float(dfm[f])
                            by += float(dfm[f + 1])
                        f += 2
                        M = mats[bi]
                        acc[0] += wt * (M[0, 0] * bx + M[0, 1] * by + M[0, 2])
                        acc[1] += wt * (M[1, 0] * bx + M[1, 1] * by + M[1, 2])
                    world[k] = acc
            tris = np.asarray(att["triangles"], dtype=np.int64).reshape(-1, 3)
            return world, uvs, tris, att
        raise ValueError(f"Unsupported attachment type {typ!r}")
