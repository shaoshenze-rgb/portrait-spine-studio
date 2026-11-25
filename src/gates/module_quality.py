"""Module QC from rendered geometry and source pixels, not attachment labels.

These engineering limits are versioned guards, not calibrated image-quality
scores. Semantic completeness and exposed fill appearance still require visual
review. A declared effect group may have disconnected visible pieces.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from src.pipeline.split_parts import _components
from src.render.renderer import raster_uv, sample_texture
from src.gates.review_binding import resolve_evidence_path, read_evidence_mask, read_evidence_rgba

RULES_VERSION = 'module-qc-10'
LIMITS = {
    # Float64 measurement tolerance, not permission for physical stretch.
    'rigid_singular_deviation': 1e-6,
    'rigid_distance_relative_error': 1e-6,
    'rigid_shear': 1e-6,
    'rigid_residual_px_at1080': 1e-4,
    'soft_singular_deviation': 0.08,
    'soft_area_deviation': 0.12,
    'soft_adjacent_rotation_degrees': 12.0,
    'soft_rotation_spread_degrees': 25.0,
    'required_motion_px_at1080': 1.0,
    'relative_motion_px_at1080': 0.4,
    'static_motion_px_at1080': 0.10,
    'contact_drift_px_at1080': 1.5,
    'grip_relative_linear_deviation': 1e-6,
    'visible_rgba_error': 2,
}
BODY_ROLES = {'head', 'face', 'body', 'torso', 'chest', 'pelvis', 'hips',
              'arm', 'upper_arm', 'lower_arm', 'forearm', 'hand', 'leg',
              'upper_leg', 'lower_leg', 'thigh', 'shin', 'body_tail', 'tail_body',
              'lower_body', 'fish_tail'}
HARD_ROLES = {'head', 'face', 'body', 'torso', 'pelvis', 'arm', 'upper_arm',
              'forearm', 'hand', 'upper_leg', 'lower_leg', 'leg', 'foot',
              'lower_body', 'weapon', 'armor'}
STATIC_ROLES = {'ground', 'pet', 'effect_group', 'prop'}
VALID_STATUSES = {'pass', 'fail', 'pending', 'not_applicable'}


def result(status: str, **kwargs: Any) -> Dict[str, Any]:
    return {'status': status, 'passed': status == 'pass', **kwargs}


def aggregate(checks: Dict[str, Any]) -> str:
    states = [c.get('status', 'pending') for c in checks.values()]
    if 'fail' in states:
        return 'fail'
    if any(s not in VALID_STATUSES or s == 'pending' for s in states):
        return 'pending'
    return 'pass' if 'pass' in states else 'not_applicable'


def recompute_quality(report: Dict[str, Any]) -> Dict[str, Any]:
    """Single aggregation path. It does not authenticate a reviewer or an input."""
    quality = report['quality']
    names = ('inputOwnership', 'bindingMotion', 'machineQC')
    states = [quality.get(k, {}).get('status', 'pending') for k in names]
    visual = quality.get('visual', {}).get('status', 'pending')
    applicable = quality.get('contractApplicable') is True
    if 'fail' in states or visual == 'fail':
        status = 'fail'
    elif (applicable and all(s in ('pass', 'not_applicable') for s in states)
          and visual == 'pass'):
        status = 'pass'
    else:
        status = 'pending'
    quality['overall'] = {'status': status, 'exportAllowed': status == 'pass'}
    return report


def finalize_visual_review(report: Dict[str, Any], *, decision: str, evidence: Dict[str, Any]) -> Dict[str, Any]:
    """Attach an explicit review after the caller verifies job/input/artifact identity."""
    if decision not in ('pass', 'fail', 'pending') or not isinstance(evidence, dict):
        raise ValueError('Invalid visual review decision/evidence')
    out = copy.deepcopy(report)
    q = out['quality']
    if decision == 'pass':
        if (q.get('contractApplicable') is not True
                or any(q.get(k, {}).get('status') not in ('pass', 'not_applicable')
                       for k in ('inputOwnership', 'bindingMotion', 'machineQC'))):
            raise ValueError('Machine/input/binding review is not complete; visual approval cannot override it')
        binding = evidence.get('review_binding', {})
        if (evidence.get('full_loop_observed') is not True
                or not evidence.get('reviewer') or not evidence.get('notes')
                or any(not binding.get(k) for k in ('job_id', 'input_sha256', 'analysis_sha256', 'artifact_sha256'))):
            raise ValueError('Visual approval needs full-loop evidence and an exact review binding')
    q['visual'] = {'status': decision, 'required': True, 'evidence': copy.deepcopy(evidence),
                   'evidenceType': 'explicit_review_declaration_not_host_attestation'}
    return recompute_quality(out)


def fit_affine(reference: np.ndarray, current: np.ndarray) -> Tuple[np.ndarray, float]:
    if len(reference) < 3 or reference.shape != current.shape:
        raise ValueError('At least three corresponding points are required')
    origin=reference.mean(axis=0);target_origin=current.mean(axis=0)
    x=reference-origin;y=current-target_origin
    if np.linalg.matrix_rank(x) < 2:
        raise ValueError('Degenerate reference geometry')
    coef = np.linalg.lstsq(x, y, rcond=None)[0]
    matrix = np.eye(3)
    matrix[:2, :2] = coef.T
    matrix[:2,2]=target_origin-origin@coef
    residual = float(np.linalg.norm(x @ coef - y, axis=1).max(initial=0.0))
    return matrix, residual


def apply_affine(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.asarray(points) @ matrix[:2, :2].T + matrix[:2, 2]


def rigidity_metrics(reference: np.ndarray, current: np.ndarray) -> Dict[str, float]:
    matrix, residual = fit_affine(reference, current)
    a = matrix[:2, :2]
    sv = np.linalg.svd(a, compute_uv=False)
    denom = max(1e-12, float(np.linalg.norm(a[:, 0]) * np.linalg.norm(a[:, 1])))
    shear = abs(float(a[:, 0] @ a[:, 1])) / denom
    # Include distances across separate visible pieces; sample deterministically, not by component.
    ids = np.unique(np.linspace(0, len(reference)-1, min(96, len(reference))).astype(int))
    r, c = reference[ids], current[ids]
    d0 = np.linalg.norm(r[:, None] - r[None, :], axis=2)
    d1 = np.linalg.norm(c[:, None] - c[None, :], axis=2)
    keep = d0 > 1e-3
    distance_error = float(np.max(np.abs(d1[keep]/d0[keep]-1))) if keep.any() else math.inf
    return {'singularDeviation': float(np.max(np.abs(sv-1))), 'shear': shear,
            'distanceRelativeError': distance_error, 'affineResidualPx': residual,
            'determinant': float(np.linalg.det(a))}


def soft_deformation_metrics(reference: np.ndarray, current: np.ndarray, triangles: np.ndarray) -> Dict[str, float]:
    maximum_strain = maximum_area = 0.0
    folded = degenerate = 0
    angles: Dict[int, float] = {}
    edges: Dict[Tuple[int, int], List[int]] = {}
    for i, tri in enumerate(np.asarray(triangles, dtype=int)):
        r, c = reference[tri], current[tri]
        b0, b1 = (r[1:]-r[0]).T, (c[1:]-c[0]).T
        det0 = float(np.linalg.det(b0))
        if abs(det0) < 1e-8:
            continue
        f = b1 @ np.linalg.inv(b0)
        det = float(np.linalg.det(f))
        if det <= 0:
            folded += 1
        if abs(det) < 1e-6:
            degenerate += 1
        u, sv, vt = np.linalg.svd(f)
        maximum_strain = max(maximum_strain, float(np.max(np.abs(sv-1))))
        maximum_area = max(maximum_area, abs(det-1))
        rotation = u @ vt
        angles[i] = math.atan2(rotation[1, 0], rotation[0, 0])
        for a, b in zip(tri, np.roll(tri, -1)):
            edges.setdefault(tuple(sorted((int(a), int(b)))), []).append(i)
    angle_diff = lambda a, b: abs(math.atan2(math.sin(a-b), math.cos(a-b))) * 180/math.pi
    adjacent = max((angle_diff(angles[a], angles[b]) for refs in edges.values()
                    for a in refs for b in refs if a != b and a in angles and b in angles), default=0.0)
    values = list(angles.values())
    mean = math.atan2(sum(math.sin(v) for v in values), sum(math.cos(v) for v in values)) if values else 0.0
    spread = max((angle_diff(a, mean) for a in values), default=0.0) * 2
    return {'maxSingularDeviation': maximum_strain, 'maxAreaDeviation': maximum_area,
            'foldedTriangles': folded, 'degenerateTriangles': degenerate,
            'maxAdjacentRotationDegrees': adjacent, 'rotationSpreadDegrees': spread}


def _enclosed(alpha: np.ndarray) -> np.ndarray:
    # Reuse the existing connectivity primitive; the motion compensation below is independent.
    from .audit import enclosed_transparent
    return enclosed_transparent(alpha, 128)


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:] |= mask[:-1]; out[:-1] |= mask[1:]
    out[:, 1:] |= mask[:, :-1]; out[:, :-1] |= mask[:, 1:]
    return out


def transport_reference_mask(mask: np.ndarray, matrix: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Inverse-map pixel centres of a setup mask. No dilation or connected exemption."""
    out = np.zeros(shape, dtype=bool)
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return out
    corners = np.array([[xs.min(), ys.min()], [xs.max()+1, ys.min()],
                        [xs.max()+1, ys.max()+1], [xs.min(), ys.max()+1]], dtype=float)
    dest = apply_affine(corners, matrix)
    x0, y0 = np.maximum(0, np.floor(dest.min(axis=0)).astype(int))
    x1, y1 = np.minimum([shape[1], shape[0]], np.ceil(dest.max(axis=0)).astype(int))
    inv = np.linalg.inv(matrix)
    for row in range(int(y0), int(y1), 128):
        yy, xx = np.mgrid[row:min(row+128, y1), x0:x1]
        q = apply_affine(np.column_stack((xx.ravel()+.5, yy.ravel()+.5)), inv)
        ix, iy = np.floor(q[:, 0]).astype(int), np.floor(q[:, 1]).astype(int)
        valid = (ix >= 0) & (iy >= 0) & (ix < mask.shape[1]) & (iy < mask.shape[0])
        values = np.zeros(len(ix), bool); values[valid] = mask[iy[valid], ix[valid]]
        out[row:min(row+128, y1), x0:x1] = values.reshape(xx.shape)
    return out


def transport_original_holes(mask: np.ndarray, source_alpha: np.ndarray, matrix: np.ndarray,
                             shape: Tuple[int,int], kernel: str='cubic', ss: int=1) -> np.ndarray:
    """Warp the *source* negative-space alpha field, including its sampling contour.

    This is not an expected rendering of the whole current scene: only holes
    registered before animation are transported. New transparent pixels and
    current-frame connectivity never enter this reference field.
    """
    out=np.zeros(shape,bool);ys,xs=np.nonzero(mask)
    if not len(xs):return out
    y0,y1=max(0,int(ys.min())-3),min(mask.shape[0],int(ys.max())+4)
    x0,x1=max(0,int(xs.min())-3),min(mask.shape[1],int(xs.max())+4)
    contour=_dilate4(_dilate4(mask)) & (source_alpha>=128) & (source_alpha<255)
    reference=np.where(mask|contour,255-source_alpha,0)[y0:y1,x0:x1].astype(np.float32)
    tex=np.repeat(reference[...,None],4,axis=2)
    corners=apply_affine(np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]],float),matrix)
    dx0,dy0=np.maximum([0,0],np.floor(corners.min(axis=0)).astype(int))
    dx1,dy1=np.minimum([shape[1],shape[0]],np.ceil(corners.max(axis=0)).astype(int))
    inv=np.linalg.inv(matrix)
    for row in range(int(dy0),int(dy1),96):
        yy,xx=np.mgrid[row:min(row+96,dy1),dx0:dx1];negative=np.zeros(xx.shape,np.float64)
        for sy in range(ss):
            for sx in range(ss):
                p=np.column_stack((xx.ravel()+(sx+.5)/ss,yy.ravel()+(sy+.5)/ss))
                q=apply_affine(p,inv)
                values=sample_texture(tex,q[:,0]-x0-.5,q[:,1]-y0-.5,kernel)[:,3]
                negative+=np.clip(values,0,255).reshape(xx.shape)/(ss*ss)
        out[row:min(row+96,dy1),dx0:dx1]=np.rint(255-negative)<128
    return out


class SoftMaterialHoles:
    """Original single-piece holes transported by existing material triangles.

    No nearest-bone extrapolation is allowed. Every original negative pixel must
    be inside the existing setup mesh and its boundary must belong to that piece.
    Large holes whose empty cells were omitted from the mesh remain unsupported.
    """
    def __init__(self, renderer: Any, pieces: Dict[str,Any], names: Sequence[str],
                 source_alpha: np.ndarray, parts: Dict[str,Any], pad: int):
        self.r=renderer;self.pieces=pieces;self.names=list(names);self.source_alpha=source_alpha
        self.parts=parts;self.pad=pad;self.registered={};self.fields={};self.failures=[]

    def register(self, mask: np.ndarray, boundary: np.ndarray)->bool:
        yy,xx=np.nonzero(mask);points=np.column_stack((xx+.5,yy+.5));eligible=[]
        for name in self.names:
            pd=self.pieces[name];part=self.parts.get(name)
            if part is None or not np.all(pd['declaredMask'][boundary]):continue
            x,y,w,h=map(int,part['bbox']);tex=self.r.tex[name];th,tw=tex.shape[:2]
            if (tw,th)!=(w+2*self.pad,h+2*self.pad):continue
            # UV packing is checked against setup image coordinates, not current alpha.
            source_points=pd['uv']*np.array([tw,th])+[x-self.pad,y-self.pad]
            error=float(np.linalg.norm(source_points-pd['reference'],axis=1).max(initial=0))
            tolerance=.001+max(tw,th)*1e-6  # Recorded UV(6d)/local coordinate packing error.
            if error>tolerance:continue
            indices,_=_point_bind(points,pd['reference'],pd['triangles'])
            if np.any(indices[:,0]<0):continue
            eligible.append(name)
        if len(eligible)!=1:
            self.failures.append('single-piece material coverage absent or ambiguous');return False
        name=eligible[0]
        if name not in self.registered:self.registered[name]=np.zeros_like(mask)
        self.registered[name]|=mask
        return True

    def prepare(self)->None:
        for name,mask in self.registered.items():
            pd=self.pieces[name];x,y,w,h=map(int,self.parts[name]['bbox']);th,tw=self.r.tex[name].shape[:2]
            contour=_dilate4(_dilate4(mask))&pd['declaredMask']&(self.source_alpha>=128)&(self.source_alpha<255)
            negative=np.where(mask|contour,255-self.source_alpha,0).astype(np.float32)
            field=np.zeros((th,tw,4),np.float32);sx,sy=x-self.pad,y-self.pad
            x0,y0=max(0,sx),max(0,sy);x1,y1=min(self.r.width,sx+tw),min(self.r.height,sy+th)
            field[y0-sy:y1-sy,x0-sx:x1-sx]=negative[y0:y1,x0:x1,None]
            # Only reference triangles intersecting registered old-negative material.
            ny,nx=np.nonzero(field[...,3]>0);uv_pixels=pd['uv']*np.array([tw,th]);tris=pd['triangles']
            if len(nx):
                lo=uv_pixels[tris].min(axis=1);hi=uv_pixels[tris].max(axis=1)
                active=np.all(hi>=[nx.min()-2,ny.min()-2],axis=1)&np.all(lo<=[nx.max()+3,ny.max()+3],axis=1)
                self.fields[name]=(field,tris[active])

    def predict(self, posed: Dict[str,np.ndarray])->np.ndarray:
        out=np.zeros((self.r.height,self.r.width),bool);ss=self.r.ss
        accum=np.zeros((self.r.height*ss,self.r.width*ss),np.float32)
        for name,(field,tris)in self.fields.items():
            if name not in posed:continue
            pd=self.pieces[name];y,x,mask,u,v=raster_uv(posed[name],pd['uv'],tris,self.r.width,self.r.height,ss)
            if not mask.any():continue
            yy,xx=np.nonzero(mask);th,tw=field.shape[:2]
            values=sample_texture(field,u[yy,xx]*tw-.5,v[yy,xx]*th-.5,self.r.kernel)[:,3]
            # Registered holes cannot overlap semantically across pieces; max avoids
            # artificially amplifying partial opacity at a shared sampling edge.
            accum[yy+y,xx+x]=np.maximum(accum[yy+y,xx+x],np.clip(values,0,255))
        if ss>1:accum=accum.reshape(self.r.height,ss,self.r.width,ss).mean(axis=(1,3))
        out[:]=np.rint(255-accum)<128
        return out


class NegativeSpaceTracker:
    """Track original holes with verified rigid or single-piece soft material.

    Unsupported soft/cross-module holes remain pending. Current output alpha never
    creates the exemption mask. A new hole adjacent to an old hole remains new.
    """
    def __init__(self, source_alpha: np.ndarray, setup_alpha: np.ndarray, owners: np.ndarray,
                 rigid_owners: Sequence[int], *, kernel: str='cubic', ss: int=1,
                 soft_materials: Optional[Dict[int,SoftMaterialHoles]]=None, reference_verified: bool=True):
        self.source_alpha = np.asarray(source_alpha, dtype=np.uint8).copy()
        self.setup_alpha = np.asarray(setup_alpha, dtype=np.uint8).copy()
        self.kernel=kernel;self.ss=ss
        self.soft_materials=soft_materials or {};self.reference_verified=reference_verified
        self.owner_masks={int(owner):((owners==owner)&(self.source_alpha>=128))for owner in np.unique(owners)if owner>=0}
        self.owner_bounds={}
        for owner,mask in self.owner_masks.items():
            yy,xx=np.nonzero(mask)
            if len(xx):self.owner_bounds[owner]=np.array([[xx.min(),yy.min()],[xx.max()+1,yy.min()],[xx.max()+1,yy.max()+1],[xx.min(),yy.max()+1]],float)
        self.supported: List[Tuple[int, np.ndarray]] = []
        self.supported_rigid_count=0;self.supported_soft_count=0
        self.unresolved: List[np.ndarray] = []
        holes = _enclosed(self.source_alpha) & _enclosed(self.setup_alpha)
        labels = _components(holes)
        allowed = set(rigid_owners)
        grouped={}
        self.unverified_original_count=0
        for label in np.unique(labels[holes]):
            if not reference_verified:
                self.unverified_original_count+=1;continue
            component = labels == label
            boundary = _dilate4(component) & ~component & (self.source_alpha >= 128)
            who = set(int(x) for x in np.unique(owners[boundary]))
            if len(who) == 1 and next(iter(who)) in allowed:
                owner=next(iter(who));grouped.setdefault(owner,np.zeros_like(component));grouped[owner]|=component;self.supported_rigid_count+=1
            elif len(who)==1 and next(iter(who))in self.soft_materials and self.soft_materials[next(iter(who))].register(component,boundary):
                self.supported_soft_count+=1
            else:
                self.unresolved.append(component)
        self.supported=list(grouped.items())
        for material in self.soft_materials.values():material.prepare()

    def evaluate(self, frame_alpha: np.ndarray, transforms: Dict[int, np.ndarray],
                 motion_bound: float = 0.0, posed_pieces: Optional[Dict[str,np.ndarray]]=None) -> Dict[str, Any]:
        predicted = np.zeros_like(frame_alpha, dtype=bool)
        current = _enclosed(frame_alpha)
        occlusion_ambiguous=np.zeros_like(predicted);coverage_cache={}
        unresolved = list(self.unresolved)
        grouped={}
        for owner,mask in self.supported:
            grouped.setdefault(owner,np.zeros_like(mask))
            grouped[owner]|=mask
        predictions={}
        for owner, mask in grouped.items():
            if owner not in transforms:
                unresolved.append(mask); continue
            predictions[owner]=transport_original_holes(mask,self.source_alpha,transforms[owner],frame_alpha.shape,self.kernel,self.ss)
        for owner,material in self.soft_materials.items():
            if material.fields:
                if posed_pieces is None:
                    unresolved.extend(material.registered.values())
                else:predictions[owner]=material.predict(posed_pieces)
        for owner,old in predictions.items():
            predicted|=old
            yy,xx=np.nonzero(old&current)
            if not len(xx):continue
            lo=np.array([xx.min(),yy.min()]);hi=np.array([xx.max()+1,yy.max()+1])
            for other,bounds in self.owner_bounds.items():
                if other==owner or other not in transforms:continue
                moved=apply_affine(bounds,transforms[other])
                if np.any(moved.max(axis=0)<=lo)or np.any(moved.min(axis=0)>=hi):continue
                if other not in coverage_cache:coverage_cache[other]=transport_reference_mask(self.owner_masks[other],transforms[other],frame_alpha.shape)
                # Another module now sweeps over this hole: its new missing material
                # cannot inherit the old-hole exemption. Layer/soft ambiguity is pending.
                occlusion_ambiguous|=old&current&coverage_cache[other]
        candidates = current & ~predicted
        # Old source-transparent locations do not seed a flood-fill exclusion.
        loss = candidates & (self.source_alpha >= 250)
        uncertain = (candidates & (self.source_alpha < 250))|occlusion_ambiguous
        if unresolved and current.any():
            labels = _components(current)
            for old in unresolved:
                ys, xs = np.nonzero(old)
                if not len(xs):
                    continue
                pad = int(math.ceil(max(0, motion_bound))) + 1
                y0,y1=max(0,int(ys.min())-pad),min(old.shape[0],int(ys.max())+pad+1)
                x0,x1=max(0,int(xs.min())-pad),min(old.shape[1],int(xs.max())+pad+1)
                labels_here = np.unique(labels[y0:y1,x0:x1][current[y0:y1,x0:x1]])
                for label in labels_here:
                    region = (labels == label) & candidates
                    uncertain |= region; loss &= ~region
        status = 'fail' if loss.any() else ('pending' if uncertain.any() or unresolved or not self.reference_verified else 'pass')
        return result(status, newHolePixels=int(loss.sum()),
                      predictedOldHolePixels=int((current & predicted).sum()),
                      ambiguousPixels=int(uncertain.sum()),
                      occlusionAmbiguousPixels=int(occlusion_ambiguous.sum()),
                      unresolvedOriginalHoles=len(unresolved)+self.unverified_original_count,
                      supportedOriginalHoles=self.supported_rigid_count+self.supported_soft_count,
                      supportedRigidHoles=self.supported_rigid_count,supportedSoftMaterialHoles=self.supported_soft_count,
                      referenceVerified=self.reference_verified,
                      scope='enclosed holes only; contacts/sweeps check open seams separately')


def _point_bind(points: np.ndarray, vertices: np.ndarray, triangles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    indices = np.full((len(points), 3), -1, dtype=int)
    weights = np.zeros((len(points), 3), dtype=float)
    for tri in triangles:
        a,b,c = vertices[tri]
        mat = np.column_stack((b-a,c-a))
        if abs(np.linalg.det(mat)) < 1e-10:
            continue
        missing = np.flatnonzero(indices[:, 0] < 0)
        if not len(missing):
            break
        q = (points[missing]-a) @ np.linalg.inv(mat).T
        w = np.column_stack((1-q.sum(axis=1),q))
        inside = np.all(w >= -1e-7, axis=1)
        indices[missing[inside]] = tri; weights[missing[inside]] = w[inside]
    return indices, weights


def _visible_samples(uv: np.ndarray, triangles: np.ndarray, texture: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    a = texture[..., 3]
    opaque = a >= .5
    ys,xs = np.nonzero(opaque)
    if not len(xs):
        return np.zeros((0,3),int),np.zeros((0,3))
    boundary = opaque & ~(_dilate4(~opaque))
    by,bx = np.nonzero(opaque & ~boundary)
    pick = np.unique(np.linspace(0,len(xs)-1,min(48,len(xs))).astype(int))
    points = np.column_stack(((xs[pick]+.5)/a.shape[1],(ys[pick]+.5)/a.shape[0]))
    if len(bx):
        pick = np.unique(np.linspace(0,len(bx)-1,min(48,len(bx))).astype(int))
        points = np.vstack((points,np.column_stack(((bx[pick]+.5)/a.shape[1],(by[pick]+.5)/a.shape[0]))))
    idx,weights = _point_bind(points,uv,triangles)
    good = idx[:,0]>=0
    return idx[good],weights[good]


def _render_slot_crop(renderer: Any, slot: Dict[str, Any], vertices: np.ndarray, bbox: Sequence[int]) -> np.ndarray:
    _,uv,tris,_ = renderer.sk.attachment_vertices(slot)
    bx,by,bw,bh=map(int,bbox);out=np.zeros((bh,bw,4),np.uint8)
    # Clip rasterization to the requested audit patch. In particular, checking
    # an exposed limb strip must not rasterize the entire clothing layer again.
    y,x,mask,u,v = raster_uv(vertices-np.array([bx,by]),uv,tris,bw,bh)
    if mask.any():
        yy,xx=np.nonzero(mask);tex=renderer.tex[slot['name']];th,tw=tex.shape[:2]
        values=sample_texture(tex,u[yy,xx]*tw-.5,v[yy,xx]*th-.5,renderer.kernel)
        out[yy+y,xx+x]=np.clip(np.rint(values),0,255).astype(np.uint8)
    return out


def _actual_skin_influences(renderer: Any, slot: Dict[str,Any], att: Dict[str,Any]) -> set:
    if att.get('type','region')!='mesh' or len(att.get('vertices',[]))==len(att.get('uvs',[])):
        return {slot['bone']}
    data=att.get('vertices',[]);names=[b.name for b in renderer.sk.bones];used=set();i=0
    while i<len(data):
        count=int(data[i]);i+=1
        for _ in range(count):
            bi,weight=int(data[i]),float(data[i+3]);i+=4
            if weight>0 and 0<=bi<len(names):used.add(names[bi])
    return used


def locked_root_runtime_error(declaration: Dict[str, Any], by_name: Dict[str, Any]) -> Optional[str]:
    """Return a reason if driver/soft chain is missing or mis-parented in the live skeleton."""
    driver = declaration.get('bone')
    soft = declaration.get('softBones')
    if not isinstance(soft, list) or not soft:
        return 'empty softBones'
    if driver not in by_name:
        return 'missing driver'
    for index, name in enumerate(soft):
        if name not in by_name:
            return 'missing '+str(name)
        parent = by_name[name].parent
        expected = driver if index == 0 else soft[index - 1]
        actual = None if parent is None else parent.name
        if actual != expected:
            return 'bad parent '+str(name)
    return None


class ModuleQualityAuditor:
    """Collect actual posed vertices alongside the already-rendered audit frames."""
    def __init__(self, renderer: Any, parts_dir: Path, rig_report: Dict[str, Any], setup_frame: np.ndarray):
        self.r=renderer; self.pdir=Path(parts_dir); self.report=rig_report
        self.evidence_root=Path(renderer.rig_dir).resolve().parent
        self.contract=rig_report.get('moduleContract'); self.applicable=isinstance(self.contract,dict) and self.contract.get('version')==1
        self.scale=1.0;self.body_height_px=None; self.source=setup_frame.copy(); self.setup_reference=setup_frame.copy();self.setup_alpha=setup_frame[...,3].copy()
        self.modules:Dict[str,Any]={}; self.pieces:Dict[str,Any]={}; self.errors=[];self.pending=[]
        self.checks_input={};self.checks_binding={};self.checks_machine={}
        self.samples=[]; self.hole_samples=[]; self.contact_samples:Dict[str,List[float]]={};self.contact_gap_samples:Dict[str,List[int]]={}
        self.grip_pose_samples:Dict[str,List[Dict[str,Any]]]={}
        self.static_invalid=[];self.hidden_risks=[];self.module_order=[];self.owner_map=np.full(self.setup_alpha.shape,-1,np.int16)
        self.material_changed=set();self.uv_drift=0.0;self.foreign_influences=[];self.uniform_soft_warp=set()
        self.remaining_hidden=[]
        self.hidden_joints=[]
        self.hidden_attachments=[]
        self.locked_root_rotations={}
        self.locked_root_invalid=set()
        if not self.applicable:
            return
        try:
            self._initialize()
        except (ValueError,KeyError,TypeError,OSError,IndexError) as exc:
            self.errors.append(str(exc))

    def _mask(self,value:Any)->np.ndarray:
        if isinstance(value,np.ndarray):
            a=value
        else:
            path=resolve_evidence_path(value,root=self.evidence_root,base=self.evidence_root)
            with Image.open(path) as image:a=np.asarray(image.convert('L'))
        if a.shape!=self.setup_alpha.shape:raise ValueError('Module mask has wrong canvas dimensions')
        return a>0

    def _initialize(self)->None:
        for m in (self.contract or {}).get('modules') or []:
            if m.get('lockedRoot'):
                reason=locked_root_runtime_error(m, self.r.sk.by_name)
                if reason:
                    ident=m.get('id')
                    self.errors.append('lockedRoot chain invalid: '+str(ident)+': '+reason)
                    self.locked_root_invalid.add(ident)
        pj_path=self.pdir/'parts.json'
        self.parts_json={}
        if pj_path.is_file():
            self.parts_json=json.loads(pj_path.read_text(encoding='utf-8'))
            source=Path(self.parts_json['source'])
            with Image.open(source) as image:self.source=np.asarray(image.convert('RGBA'),dtype=np.uint8)
            if self.source.shape[:2]!=self.setup_alpha.shape:raise ValueError('Source and canvas dimensions differ')
            digest=hashlib.sha256(source.read_bytes()).hexdigest()
            refs={'parts':self.parts_json.get('sourceSHA256'),'contract':self.contract.get('sourceSHA256')}
            mismatch=[k for k,v in refs.items()if v is not None and v!=digest]
            if self.report.get('partsJsonSHA256')and self.report['partsJsonSHA256']!=hashlib.sha256(pj_path.read_bytes()).hexdigest():mismatch.append('parts_manifest')
            self.checks_input['sourceIdentity']=result('fail'if mismatch else('pending'if any(v is None for v in refs.values())else'pass'),
                actualSourceSHA256=digest,mismatchedBindings=mismatch,missingBindings=[k for k,v in refs.items()if v is None])
        else:self.pending.append('Source parts manifest missing')
        modules=self.contract.get('modules')
        if not isinstance(modules,list) or not modules:raise ValueError('Module contract is empty')
        self.r.pose(None)
        slots={s['name']:s for s in self.r.sk.slots}
        seen=set();claimed=np.zeros(self.setup_alpha.shape,bool);conflicts=0;empty_modules=[]
        required=self.contract.get('requiredModules')
        if not isinstance(required,list) or not required:self.pending.append('requiredModules is missing')
        for owner,m in enumerate(modules):
            ident=m.get('id')
            if not isinstance(ident,str)or ident in self.modules:raise ValueError('Invalid/duplicate module ID')
            if m.get('shape') not in ('rigid','soft'):raise ValueError('Module shape must be rigid or soft')
            if m.get('motion') not in ('required','static','optional'):raise ValueError('Module motion declaration missing')
            names=m.get('pieces')
            if not isinstance(names,list)or not names:raise ValueError('Module has no pieces: '+ident)
            owned=np.zeros(self.setup_alpha.shape,bool)
            masks=m.get('visibleMasks')
            if not isinstance(masks,list)or not masks:self.pending.append('Visible reference masks missing: '+ident)
            else:
                for v in masks:owned|=self._mask(v)
            owned &= self.source[...,3]>0
            if not owned.any():empty_modules.append(ident)
            conflicts+=int((claimed&owned).sum());claimed|=owned;self.owner_map[owned]=owner
            info={'declaration':m,'owner':owner,'mask':owned,'references':[],'tracks':[],
                  'rigidMax':{},'softMax':{},'softMaxNormal':{},'motionMax':0.0,'pieceNames':names,
                  'boneReferences':{},'softBoneSingularDeviation':0.0,'softBoneShear':0.0}
            allowed_bones={m.get('bone')}|set(m.get('softBones',[]))
            if not m.get('bone'):self.pending.append('Module driver identity missing: '+ident)
            for name in names:
                if name in seen:raise ValueError('Piece belongs to more than one module: '+str(name))
                if name not in slots:raise ValueError('Declared piece missing from runtime: '+str(name))
                seen.add(name);slot=slots[name];world,uv,tris,att=self.r.sk.attachment_vertices(slot);vc=self.r.to_canvas(world)
                used_bones=_actual_skin_influences(self.r,slot,att)
                if m.get('bone')and used_bones-allowed_bones:self.foreign_influences.append({'module':ident,'piece':name,'bones':sorted(used_bones-allowed_bones)})
                if m['shape']=='soft':
                    for bn in used_bones|{m.get('bone')}:
                        if bn is None:continue
                        if bn not in self.r.sk.by_name:raise ValueError('Module driver not present: '+str(bn))
                        mat=self.r.sk.by_name[bn].matrix()[:,:2].copy()
                        if abs(np.linalg.det(mat))<1e-12:raise ValueError('Singular module influence: '+bn)
                        info['boneReferences'][bn]=mat
                idx,w=_visible_samples(uv,tris,self.r.tex[name])
                if not len(idx):raise ValueError('Piece has no measurable visible pixels: '+name)
                pts=(vc[idx]*w[:,:,None]).sum(axis=1)
                self.pieces[name]={'slot':slot,'reference':vc.copy(),'triangles':tris.copy(),'sampleIndices':idx,'sampleWeights':w,
                                   'referenceSamples':pts,'module':ident,
                                   'uv':uv.copy(),'textureSHA256':hashlib.sha256(memoryview(np.ascontiguousarray(self.r.tex[name]))).hexdigest(),
                                   'declaredMask':self._mask(masks[names.index(name)]) if isinstance(masks,list)and len(masks)==len(names)else owned}
                info['references'].append(vc.copy())
            info['reference']=np.vstack(info['references']);self.modules[ident]=info;self.module_order.append(ident)
        missing_required=[x for x in (required or[])if x not in self.modules]
        unowned_slots=sorted(set(slots)-seen)
        body_mask=np.zeros(self.setup_alpha.shape,bool)
        for info in self.modules.values():
            if info['declaration'].get('role')in BODY_ROLES|{'foot'}:body_mask|=info['mask']
        if not body_mask.any():body_mask=self.source[...,3]>=128
        yy,_=np.nonzero(body_mask)
        measured_height=float(yy.max()-yy.min()+1)if len(yy)else 0.0
        reported_height=self.report.get('bodyHeightPx')
        self.body_height_px=measured_height
        if self.body_height_px<=0:raise ValueError('Cannot measure actual body height')
        self.scale=self.body_height_px/1080.0
        height_matches=(reported_height is None or isinstance(reported_height,(int,float))
                        and not isinstance(reported_height,bool)and math.isfinite(reported_height)
                        and abs(float(reported_height)-measured_height)<=max(.01,measured_height*1e-6))
        self.checks_input['bodyScaleIdentity']=result('pass'if height_matches else'fail',
            measuredBodyHeightPx=measured_height,declaredBodyHeightPx=reported_height,
            thresholdSource='actual anatomical visible masks, never canvas padding or an unchecked report value')
        fg=self.source[...,3]>0;missing=int((fg&~claimed).sum())
        self.checks_input['declaredCoverage']=result('fail' if missing or conflicts or missing_required or unowned_slots or empty_modules else ('pending'if self.pending else'pass'),
            missingPixels=missing,overlappingPixels=conflicts,missingRequiredModules=missing_required,unownedRuntimePieces=unowned_slots,
            emptyVisibleModules=empty_modules,
            scope='declared mask coverage only, not proof of semantic completeness')
        mask_mismatches=[];mask_missing=[]
        declared_hashes=self.contract.get('visibleMaskSHA256',{})
        for m in modules:
            for name,value in zip(m['pieces'],m.get('visibleMasks',[])):
                expected=declared_hashes.get(name)
                if expected is None:mask_missing.append(name);continue
                if isinstance(value,np.ndarray):mask_mismatches.append(name);continue
                path=resolve_evidence_path(value,root=self.evidence_root,base=self.evidence_root)
                if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:mask_mismatches.append(name)
        for o in self.contract.get('occlusions',[]):
            if not o.get('hiddenMask'):continue
            name=o.get('back');expected=self.contract.get('hiddenMaskSHA256',{}).get(name)
            path=resolve_evidence_path(o['hiddenMask'],root=self.evidence_root,base=self.evidence_root)
            if expected is None:mask_missing.append('hidden:'+str(name))
            elif hashlib.sha256(path.read_bytes()).hexdigest()!=expected:mask_mismatches.append('hidden:'+str(name))
        self.checks_input['maskIdentity']=result('fail'if mask_mismatches else('pending'if mask_missing else'pass'),
            mismatchedBindings=mask_mismatches,missingBindings=mask_missing)
        self._check_actual_ownership()
        self._check_hidden_fill()
        self._prepare_remaining_hidden()
        foot_modules={c.get(k,{}).get('module')for c in self.contract.get('contacts',[])
                      if c.get('type')in('foot','foot_anchor')for k in('a','b')}
        for ident,info in self.modules.items():
            m=info['declaration'];role=m.get('role','')
            if m['motion']!='required' and role in BODY_ROLES:self.static_invalid.append(ident)
            if m['motion']=='static' and not m.get('staticReason') and ident not in foot_modules:self.static_invalid.append(ident)
            if m['motion']=='static'and role not in STATIC_ROLES and not(role=='foot'and ident in foot_modules):self.static_invalid.append(ident)
            if role in HARD_ROLES and m['shape']!='rigid':self.errors.append('Hard semantic role declared soft: '+ident)
            # Attached effect_group may be a locked-root soft chain; static remains allowed.
            if m.get('motion') not in ('required','static'):
                self.errors.append('motion must be required or static: '+ident)
            if m.get('lockedRoot') and m.get('motion')!='required':
                self.errors.append('lockedRoot motion must be required: '+ident)
                self.locked_root_invalid.add(ident)
            if m.get('lockedRoot') and (role!='effect_group' or m.get('shape')!='soft'
                    or m.get('motionProfile')!='soft_bend' or not m.get('parent')):
                self.errors.append('lockedRoot requires an attached effect_group soft chain: '+ident)
                self.locked_root_invalid.add(ident)
            if role=='effect_group' and m.get('shape')=='soft' and m.get('motionProfile')=='soft_bend' and not m.get('lockedRoot'):
                self.errors.append('effect_group soft chain requires lockedRoot: '+ident)
            if role=='armor' and (m.get('shape')!='rigid' or m.get('motionProfile') not in ('sway','follow')):
                self.errors.append('Armor must be rigid sway or follow: '+ident)
            if role=='weapon' and (m.get('shape')!='rigid' or m.get('motionProfile')!='follow'):
                self.errors.append('Weapon must be rigid follow: '+ident)
            if m.get('lockedRoot'):
                cur=m.get('parent'); active=False
                while cur is not None:
                    ancestor=next((info['declaration'] for info in self.modules.values() if info['declaration']['id']==cur), None)
                    if ancestor is None: break
                    if ancestor.get('motionProfile') not in ('follow','static'):
                        active=True; break
                    cur=ancestor.get('parent')
                if not active:
                    self.errors.append('lockedRoot has no active host: '+ident)
                    self.locked_root_invalid.add(ident)
                reason=locked_root_runtime_error(m, self.r.sk.by_name)
                if reason:
                    self.errors.append('lockedRoot chain invalid: '+ident+': '+reason)
                    self.locked_root_invalid.add(ident)
        self.checks_binding['motionDeclarations']=result('fail'if self.static_invalid else'pass',invalidModules=sorted(set(self.static_invalid)))
        rigid=[info['owner']for info in self.modules.values()if info['declaration']['shape']=='rigid']
        parts_by_name={p['name']:p for p in self.parts_json.get('parts',[])}
        soft_materials={info['owner']:SoftMaterialHoles(self.r,self.pieces,info['pieceNames'],self.source[...,3],parts_by_name,int(self.report.get('padPx',0)))
                        for info in self.modules.values()if info['declaration']['shape']=='soft'}
        source_pm=self.source.astype(float).copy();source_pm[...,:3]*=self.source[...,3:4]/255
        setup_pm=self.setup_reference
        setup_source_error=int(np.abs(np.rint(source_pm).astype(np.int16)-setup_pm.astype(np.int16)).max(initial=0))
        reference_verified=setup_source_error<=2 and all(self.checks_input[k]['status']=='pass'for k in('sourceIdentity','maskIdentity','declaredCoverage','actualOwnership','visibleTexture'))
        self.checks_input['negativeSpaceReference']=result('pass'if reference_verified else'pending',setupSourceMaxPremultRGBA=setup_source_error,
            scope='source/input identity and original visible material verified before old-hole registration')
        self.negative=NegativeSpaceTracker(self.source[...,3],self.setup_alpha,self.owner_map,rigid,kernel=self.r.kernel,ss=self.r.ss,
                                          soft_materials=soft_materials,reference_verified=reference_verified)
        self._prepare_contacts()

    def _check_actual_ownership(self)->None:
        path=self.pdir/'work/labels.npy'
        if not path.is_file()or not self.parts_json:
            self.checks_input['actualOwnership']=result('pending',reason='Actual source ownership labels missing');return
        labels=np.load(path); parts=self.parts_json['parts']
        if labels.shape!=self.setup_alpha.shape:raise ValueError('Actual ownership labels have wrong dimensions')
        actual=np.full(labels.shape,-1,np.int16)
        for i,p in enumerate(parts):
            name=p['name']
            if name in self.pieces:actual[labels==i]=self.modules[self.pieces[name]['module']]['owner']
        fg=self.source[...,3]>0
        mismatch=fg&(self.owner_map!=actual)
        self.checks_input['actualOwnership']=result('fail'if mismatch.any()else'pass',foreignOrMissingPixels=int(mismatch.sum()),
            connectedComponentsNotUsedAsSemanticModules=True)

    def _check_hidden_fill(self)->None:
        if not self.parts_json:
            self.checks_machine['hiddenFill']=result('pending',reason='Parts/source required');return
        allow={k:np.zeros(self.setup_alpha.shape,bool)for k in self.modules}
        for o in self.contract.get('occlusions',[]):
            if o.get('completionMode') not in ('local','provided'):
                continue  # An overlap declaration alone does not authorize hidden material.
            back_name=o.get('back');front_name=o.get('front')
            back=self.pieces[back_name]['module']if back_name in self.pieces else back_name
            front=self.pieces[front_name]['module']if front_name in self.pieces else front_name
            if back not in self.modules or front not in self.modules:continue
            region=np.zeros(self.setup_alpha.shape,bool)
            if o.get('hiddenMask') or o.get('mask'):
                region=self._mask(o.get('hiddenMask')or o.get('mask'))
            else:
                polys=o.get('polygons')or([o['polygon']]if o.get('polygon')else[])
                im=Image.new('L',(self.r.width,self.r.height));draw=ImageDraw.Draw(im)
                for poly in polys:draw.polygon([tuple(q)for q in poly],fill=255)
                region=np.asarray(im)>0
            front_mask=self.pieces[front_name]['declaredMask']if front_name in self.pieces else self.modules[front]['mask']
            allow[back]|=region&front_mask
        labels_path=self.pdir/'work/labels.npy'
        labels=np.load(labels_path)if labels_path.is_file()else None
        pollution=outside=hidden=hidden_nonzero=foreign_colour=provided_changed=0
        for idx,p in enumerate(self.parts_json['parts']):
            if p['name'] not in self.pieces:continue
            piece=self.pieces[p['name']];mid=piece['module'];vertices=piece['reference']
            # Inspect the actual attachment extent, including texture padding:
            # the old parts bbox cannot authorize material outside that bbox.
            x=max(0,int(np.floor(vertices[:,0].min())));y=max(0,int(np.floor(vertices[:,1].min())))
            right=min(self.r.width,int(np.ceil(vertices[:,0].max())))
            bottom=min(self.r.height,int(np.ceil(vertices[:,1].max())))
            w,h=max(0,right-x),max(0,bottom-y)
            part=_render_slot_crop(self.r,piece['slot'],vertices,[x,y,w,h])
            source=self.source[y:y+h,x:x+w];owned=self.modules[mid]['mask'][y:y+h,x:x+w]
            visible=owned if labels is None else(labels[y:y+h,x:x+w]==idx)
            expected=source.astype(float).copy();expected[...,:3]*=source[...,3:4]/255.0
            expected=np.rint(expected).astype(np.uint8)
            diff=np.abs(part.astype(np.int16)-expected.astype(np.int16)).max(axis=2)
            pollution+=int((visible&(source[...,3]>0)&(diff>LIMITS['visible_rgba_error'])).sum())
            extra=(part[...,3]>0)&~owned
            hidden+=int(((part[...,3]>=128)&~owned).sum())
            hidden_nonzero+=int(extra.sum());outside+=int((extra&~allow[mid][y:y+h,x:x+w]).sum())
            supplied_region=np.zeros(extra.shape,bool)
            if p.get('providedImage'):
                refs=[oc for oc in self.contract.get('occlusions',[]) if oc.get('back')==p['name']
                      and oc.get('completionMode')=='provided' and oc.get('providedImage')==p['providedImage']
                      and oc.get('providedImageSHA256')==p.get('providedImageSHA256')]
                if not refs:raise ValueError('Canvas part material has no matching declared handoff')
                _,_,supplied=read_evidence_rgba(p['providedImage'],p.get('providedImageSHA256'),
                    (self.r.width,self.r.height),root=self.evidence_root)
                supplied=supplied[y:y+h,x:x+w];supplied_region=supplied[...,3]>0
                supplied_pm=supplied.astype(float).copy();supplied_pm[...,:3]*=supplied[...,3:4]/255.0
                delta=np.abs(part.astype(int)-np.rint(supplied_pm).astype(int)).max(axis=2)
                provided_changed+=int((supplied_region&(delta>LIMITS['visible_rgba_error'])).sum())
            if extra.any()and p.get('completionMode')in('local','provided'):
                known=self.source[self.modules[mid]['mask']]
                known=known[known[:,3]>0,:3].astype(float)
                opaque_extra=extra&(part[...,3]>=250)&~supplied_region
                if opaque_extra.any() and len(known):
                    rgb=part[opaque_extra,:3].astype(float)*255/part[opaque_extra,3:4]
                    foreign_colour+=int(np.any((rgb<known.min(axis=0)-2)|(rgb>known.max(axis=0)+2),axis=1).sum())
        if hidden_nonzero:self.hidden_risks.append('Hidden material exists; exposed completion appearance needs independent full-loop review')
        self.checks_input['visibleTexture']=result('fail'if pollution else'pass',changedOwnedVisiblePixels=pollution,threshold=LIMITS['visible_rgba_error'],
                                                 measurement='actual rig texture raster versus source premultiplied RGBA, including semitransparency')
        self.checks_machine['hiddenFill']=result('fail'if outside or foreign_colour or provided_changed else'pass',hiddenOpaquePixels=hidden,undeclaredHiddenPixels=outside,
                                               hiddenNonzeroAlphaPixels=hidden_nonzero,
                                               localFillOutsideOwnSourceColourBounds=foreign_colour,
                                               changedCanvasMaterialPixels=provided_changed,
                                               appearanceApproval='pending_visual_review'if hidden_nonzero else'not_applicable')

    def _prepare_remaining_hidden(self)->None:
        """Register omitted source material, never infer its expectation from frames."""
        # A cloth and anatomical limb can be siblings without any connection
        # between them. Require their touching layer boundary to be reviewed,
        # rather than interpreting an empty occlusion list as complete coverage.
        declared={(o.get('front'),o.get('back')) for o in self.contract.get('occlusions',[])
                  if not o.get('autoSeamUnderlap')}
        candidates=[]
        ordered=[s['name'] for s in self.r.sk.slots if s['name'] in self.pieces]
        for i,back_name in enumerate(ordered):
            back=self.pieces[back_name];bm=self.modules[back['module']]['declaration']
            if bm.get('shape')!='rigid' or bm.get('role') not in BODY_ROLES or not bm.get('parent'):continue
            frontier=_dilate4(back['declaredMask'])
            for front_name in ordered[i+1:]:
                front=self.pieces[front_name];fm=self.modules[front['module']]['declaration']
                if (fm.get('shape')!='soft' or fm.get('role')=='effect_group'
                        or fm.get('parent')!=bm['parent'] or front['module']==back['module']
                        or (front_name,back_name) in declared):continue
                boundary=frontier&front['declaredMask']
                if boundary.any():
                    candidates.append({'front':front_name,'back':back_name,'parent':bm['parent'],
                                       'touchingPixels':int(boundary.sum())})
        self.checks_machine['occlusionDeclarations']=result('pending' if candidates else 'pass',
            undeclaredSiblingBoundaries=candidates,
            scope='touching anatomical rigid-back / soft-front siblings need explicit occlusion planning; candidates are not inferred hidden material and never authorize auto fill')
        material_cache={}
        for o in self.contract.get('occlusions',[]):
            # Auto contact underlaps are not anatomical occlusion declarations.
            # Every explicit declaration, including none/subset completion, must
            # register its whole expected hidden domain, not only derived bands.
            if o.get('autoSeamUnderlap'):continue
            band=o.get('bandDerivation') or {}
            if not isinstance(band,dict):raise ValueError('Invalid hidden band metadata')
            entry={'front':o.get('front'),'back':o.get('back'),'samples':[],
                   'unfilledEligiblePixels':band.get('unfilledEligiblePixels'),
                   'unsupportedTranslucentPixels':band.get('unsupportedTranslucentPixels')}
            self.remaining_hidden.append(entry)
            if (o.get('sourceHiddenTargetRegistration')!='original-back-claim-and-occlusion-and-visible-front'
                    or any(not o.get(k)for k in('sourceHiddenTargetMask','sourceHiddenTargetMaskSHA256',
                                                'remainingHiddenMask','remainingHiddenMaskSHA256'))):
                entry.update(status='pending',reason='Source-registered remaining hidden material is missing');continue
            registered={}
            for key in('sourceHiddenTargetMask','remainingHiddenMask'):
                try:
                    _,_,registered[key]=read_evidence_mask(o[key],o[key+'SHA256'],
                        (self.r.width,self.r.height),root=self.evidence_root,base=self.evidence_root)
                except ValueError as exc:
                    entry.update(status='fail',reason='Registered hidden coverage mask invalid: '+str(exc));break
            if entry.get('status'):continue
            front=self.pieces.get(o.get('front'));back=self.pieces.get(o.get('back'))
            if front is None or back is None:
                entry.update(status='pending',reason='Declared hidden coverage pieces are missing');continue
            target=registered['sourceHiddenTargetMask'];remaining=registered['remainingHiddenMask']
            filled=np.zeros_like(target)
            for other in self.contract.get('occlusions',[]):
                if other.get('back')==o.get('back')and other.get('completionMode')in('local','provided')and other.get('hiddenMask'):
                    filled|=self._mask(other['hiddenMask'])
            if (not np.array_equal(remaining,target&~filled)
                    or (target&~front['declaredMask']).any()
                    or (target&back['declaredMask']).any()
                    or (target&(self.source[...,3]==0)).any()
                    or o.get('remainingHiddenPixels')!=int(remaining.sum())):
                entry.update(status='fail',reason='Remaining material disagrees with registered target, front or fill');continue
            entry['registeredRemainingPixels']=int(remaining.sum())
            # A generated mask is a request, not proof of opaque material. The
            # splitter may fade its last pixels or fail to render them in a mesh.
            actual_remaining=target.copy()
            yy,xx=np.nonzero(target)
            if len(xx):
                x,y=int(xx.min()),int(yy.min());right,bottom=int(xx.max())+1,int(yy.max())+1
                rendered=_render_slot_crop(self.r,back['slot'],back['reference'],[x,y,right-x,bottom-y])
                actual_remaining[y:bottom,x:right]&=~(filled[y:bottom,x:right]&(rendered[...,3]>=250))
            remaining=actual_remaining
            entry['remainingPixels']=int(remaining.sum())
            if any(self.checks_input[k]['status']!='pass'for k in('sourceIdentity','maskIdentity','declaredCoverage','actualOwnership')):
                entry.update(status='pending',reason='Source and mask identity must pass before coverage can be proved');continue
            if not remaining.any():
                entry.update(status='pass',reason='No omitted material in this registered target');continue
            if self.modules[back['module']]['declaration']['shape']!='rigid':
                entry.update(status='pending',reason='Remaining hidden region lacks supported soft material coordinates');continue
            declared_front_modules={self.pieces[v['front']]['module']for v in self.contract.get('occlusions',[])
                                    if v.get('back')==o.get('back')and v.get('front')in self.pieces}
            slot_order={v['name']:i for i,v in enumerate(self.r.sk.slots)}
            occluders=[];unsupported=[]
            for name,piece in self.pieces.items():
                if piece['module']not in declared_front_modules or slot_order[name]<=slot_order[o['back']]:continue
                module=self.modules[piece['module']]
                if module['declaration']['shape']=='soft':
                    # Test actual deformed front material, never its driver-only
                    # affine approximation or the composited scene's alpha.
                    occluders.append({'piece':name,'soft':True});continue
                if name in material_cache:
                    occluders.append(material_cache[name]);continue
                visible=piece['declaredMask'].copy();opaque=visible&(self.source[...,3]==255)
                # Declared local material of the occluder can also cover a lower
                # layer, but only where it was registered and actually generated.
                hidden=np.zeros_like(visible)
                for front_fill in self.contract.get('occlusions',[]):
                    if front_fill.get('back')==name and front_fill.get('completionMode')in('local','provided')and front_fill.get('hiddenMask'):
                        hidden|=self._mask(front_fill['hiddenMask'])
                if hidden.any():
                    vertices=piece['reference'];x=max(0,int(np.floor(vertices[:,0].min())));y=max(0,int(np.floor(vertices[:,1].min())))
                    right=min(self.r.width,int(np.ceil(vertices[:,0].max())));bottom=min(self.r.height,int(np.ceil(vertices[:,1].max())))
                    rendered=_render_slot_crop(self.r,piece['slot'],vertices,[x,y,right-x,bottom-y])
                    region=hidden[y:bottom,x:right]
                    visible[y:bottom,x:right]|=region&(rendered[...,3]>0)
                    opaque[y:bottom,x:right]|=region&(rendered[...,3]==255)
                material_cache[name]={'piece':name,'owner':module['owner'],'visible':visible,'opaque':opaque}
                occluders.append(material_cache[name])
            yy,xx=np.nonzero(remaining)
            entry.update(points=np.column_stack((xx+.5,yy+.5)),backOwner=self.modules[back['module']]['owner'],
                         opaqueSource=self.source[yy,xx,3]==255,occluders=occluders,
                         supportedOccluderPieces=[v['piece']for v in occluders],unsupportedOccluderPieces=unsupported)

    def _observe_remaining_hidden(self,t:float,gain:float,transforms:dict)->None:
        for entry in self.remaining_hidden:
            if entry.get('status'):continue
            # Inverse-map the independently registered back material into the
            # declared front's source material; current composed alpha is not proof.
            world=apply_affine(entry['points'],transforms[entry['backOwner']])
            covered=np.zeros(len(world),bool);opaque=covered.copy()
            for occluder in entry['occluders']:
                if occluder.get('soft'):
                    piece=self.pieces[occluder['piece']]
                    vertices=self.r.to_canvas(self.r.sk.attachment_vertices(piece['slot'])[0])
                    x=max(0,int(np.floor(world[:,0].min())));y=max(0,int(np.floor(world[:,1].min())))
                    right=min(self.r.width,int(np.ceil(world[:,0].max()))+1)
                    bottom=min(self.r.height,int(np.ceil(world[:,1].max()))+1)
                    if right>x and bottom>y:
                        rendered=_render_slot_crop(self.r,piece['slot'],vertices,[x,y,right-x,bottom-y])
                        xx,yy=np.floor(world[:,0]).astype(int)-x,np.floor(world[:,1]).astype(int)-y
                        valid=(xx>=0)&(yy>=0)&(xx<right-x)&(yy<bottom-y)
                        alpha=rendered[yy[valid],xx[valid],3]
                        covered[valid]|=alpha>0;opaque[valid]|=alpha==255
                    continue
                source=apply_affine(world,np.linalg.inv(transforms[occluder['owner']]))
                xx,yy=np.floor(source[:,0]).astype(int),np.floor(source[:,1]).astype(int)
                valid=(xx>=0)&(yy>=0)&(xx<self.r.width)&(yy<self.r.height)
                covered[valid]|=occluder['visible'][yy[valid],xx[valid]]
                opaque[valid]|=occluder['opaque'][yy[valid],xx[valid]]
            exposed=(~covered)&entry['opaqueSource']
            unsupported=int(exposed.sum())if entry['unsupportedOccluderPieces']else 0
            entry['samples'].append({'time':round(t,6),'gain':gain,
                                     'exposedUnfilledPixels':0 if unsupported else int(exposed.sum()),
                                     'unsupportedCoveragePixels':unsupported,
                                     'translucentUnfilledPixels':int(((covered&~opaque)|~entry['opaqueSource']).sum())})

    def _finish_remaining_hidden(self)->dict:
        checks={}
        for index,entry in enumerate(self.remaining_hidden):
            samples=entry['samples'];exposed=max((v['exposedUnfilledPixels']for v in samples),default=0)
            translucent=max((v['translucentUnfilledPixels']for v in samples),default=0)
            unsupported=max((v['unsupportedCoveragePixels']for v in samples),default=0)
            status=entry.get('status')or('fail'if exposed else('pending'if translucent or unsupported or not samples else'pass'))
            fields={k:v for k,v in entry.items()if k not in('points','backOwner','opaqueSource','occluders','status')}
            checks[str(index)]=result(status,exposedUnfilledPixelsMax=exposed,translucentUnfilledPixelsMax=translucent,
                                     unsupportedCoveragePixelsMax=unsupported,**fields)
        return result(aggregate(checks)if checks else'not_applicable',occlusions=list(checks.values()),
                      scope='all registered hidden target pixel centres without actual opaque back fill at reported audit times; rigid back coordinates and actual declared rigid/soft front material, not composed frame alpha; no declarations is not proof of anatomical completeness')

    def _prepare_contacts(self)->None:
        self.contacts=[]
        for c in self.contract.get('contacts',[]):
            ident=c.get('id')
            if not ident or any(x['id']==ident for x in self.contacts):raise ValueError('Invalid contact ID')
            if c.get('type')not in('grip','joint','foot','foot_anchor','attachment','weld'):
                self.pending.append('Unsupported contact type: '+str(c.get('type')));continue
            entry={'id':ident,'type':c['type']}
            for key in('a','b'):
                endpoint=c.get(key)
                if endpoint is None:continue
                mid=endpoint.get('module');piece=endpoint.get('piece');point=np.asarray(endpoint.get('point'),dtype=float)
                if mid not in self.modules or piece not in self.pieces or self.pieces[piece]['module']!=mid or point.shape!=(2,)or not np.isfinite(point).all():
                    raise ValueError('Invalid contact endpoint: '+str(ident))
                pd=self.pieces[piece];inds,ws=_point_bind(point[None],pd['reference'],pd['triangles'])
                entry[key]={'module':mid,'piece':piece,'point':point,'indices':inds[0],'weights':ws[0]}
            if 'a'not in entry:raise ValueError('Contact needs endpoint a')
            if 'b'not in entry:
                world=np.asarray(c.get('world'),dtype=float)
                if world.shape!=(2,)or not np.isfinite(world).all():raise ValueError('Contact needs b or fixed world point')
                entry['world']=world
            a=entry['a']['point'];b=entry.get('b',{}).get('point',entry.get('world'));mid=(a+b)/2
            radius=max(1,int(round(2*self.scale))); yy,xx=np.mgrid[-radius:radius+1,-radius:radius+1]
            points=np.column_stack((xx.ravel()+mid[0],yy.ravel()+mid[1]));xi=np.floor(points[:,0]).astype(int);yi=np.floor(points[:,1]).astype(int)
            good=(xi>=1)&(yi>=1)&(xi<self.r.width-1)&(yi<self.r.height-1)
            interior=(self.source[...,3]>=250)&~_dilate4(self.source[...,3]<250)
            good[good]&=interior[yi[good],xi[good]]
            # Probe material pixel centres, not fixed screen points near a moving silhouette.
            entry['coveragePoints']=np.column_stack((xi[good]+.5,yi[good]+.5));entry['referenceMidpoint']=mid
            self.contacts.append(entry);self.contact_samples[ident]=[];self.contact_gap_samples[ident]=[]
            if c['type']=='grip':self.grip_pose_samples[ident]=[]
            if c.get('hiddenBy'):
                if c['type'] == 'attachment':
                    self._prepare_hidden_attachment(c)
                else:
                    self._prepare_hidden_joint(c)

    def _verified_canvas_alpha(self, name, expected_sha):
        part=next((p for p in self.parts_json.get('parts',[]) if p['name']==name),None)
        if not part or self.parts_json.get('materialMode')!='canvas_layers':
            raise ValueError('Canvas layer material manifest missing')
        layer=part.get('canvasLayer') or {}
        if layer.get('sha256')!=expected_sha:raise ValueError('Canvas occluder layer binding changed')
        _,_,rgba=read_evidence_rgba(layer.get('image'),expected_sha,(self.r.width,self.r.height),
                                   root=self.evidence_root,base=self.evidence_root)
        if not np.array_equal(rgba[...,3]>0,self.pieces[name]['declaredMask']):
            raise ValueError('Canvas support differs from actual material')
        x,y,w,h=part['bbox'];pad=int(self.report.get('padPx',0))
        expected=np.zeros((h+2*pad,w+2*pad,4),np.float32)
        crop=rgba[y:y+h,x:x+w].astype(np.float32);crop[...,:3]*=crop[...,3:4]/255.0
        expected[pad:pad+h,pad:pad+w]=crop
        if not np.array_equal(expected,self.r.tex[name]):
            raise ValueError('Runtime Canvas texture differs from registered layer')
        return rgba[...,3]

    def _prepare_hidden_attachment(self,c):
        from src.studio.module_contract import SOFT_ROLES, validate_hidden_attachment_region
        from src.studio.vision_schema import rasterize_polygons
        a,b=c['a']['module'],c['b']['module']
        if self.report.get('partsJsonSHA256')!=hashlib.sha256((self.pdir/'parts.json').read_bytes()).hexdigest():
            raise ValueError('Canvas material manifest changed after rig build')
        am,bm=self.modules[a]['declaration'],self.modules[b]['declaration']
        if (self.contract.get('materialMode')!='canvas_layers' or c['type']!='attachment'
                or bm.get('parent')!=a or am.get('shape')!='rigid' or am.get('role') not in BODY_ROLES-{'foot'}
                or bm.get('shape')!='soft' or bm.get('role') not in SOFT_ROLES-{'effect_group','pet','prop','ground'}):
            raise ValueError('Invalid hidden Canvas attachment')
        refs=c['hiddenBy'];orders={s['name']:i for i,s in enumerate(self.r.sk.slots)}
        if not isinstance(refs,list) or not 1<=len(refs)<=8:raise ValueError('Invalid attachment references')
        targets=[];fronts=[];seen=set()
        for ref in refs:
            if not isinstance(ref,dict) or set(ref)!={'front','back'}:raise ValueError('Invalid attachment pair')
            pair=(ref['front'],ref['back'])
            if pair in seen or pair[0] not in self.pieces or pair[1] not in self.pieces:
                raise ValueError('Missing or duplicate attachment layer')
            seen.add(pair);front,back=(self.pieces[n] for n in pair)
            fm=self.modules[front['module']]['declaration']
            linked=any(v.get('a',{}).get('module')==a and v.get('b',{}).get('module')==front['module']
                       for v in self.contract.get('contacts',[]))
            if (back['module']!=a or orders[pair[0]]<=orders[pair[1]]
                    or (front['module']!=b and (fm.get('parent')!=a or not linked))
                    or fm.get('role') in ('weapon','effect_group','pet','prop','foot')):
                raise ValueError('Unrelated or wrongly ordered attachment occluder')
            oc=next((o for o in self.contract.get('occlusions',[]) if (o.get('front'),o.get('back'))==pair),None)
            if not oc or oc.get('completionMode')!='none' or not oc.get('hiddenAttachment'):
                raise ValueError('Missing explicit Canvas attachment occlusion')
            if oc.get('sourceInputSHA256')!=self.contract.get('sourceSHA256'):
                raise ValueError('Attachment source binding mismatch')
            fa=self._verified_canvas_alpha(pair[0],oc.get('frontLayerSHA256'))
            ba=self._verified_canvas_alpha(pair[1],oc.get('backLayerSHA256'))
            normalized=[[[x/self.r.width,y/self.r.height] for x,y in poly] for poly in oc.get('polygons',[])]
            zone=rasterize_polygons(normalized,self.r.width,self.r.height)
            missing=zone&(ba==0)
            if not missing.any() or (missing&(fa==0)).any():
                raise ValueError('Attachment declaration omits static uncovered material')
            for key in ('sourceHiddenTargetMask','remainingHiddenMask'):
                _,_,mask=read_evidence_mask(oc.get(key),oc.get(key+'SHA256'),(self.r.width,self.r.height),
                                            root=self.evidence_root,base=self.evidence_root)
                if not np.array_equal(mask,missing):raise ValueError('Attachment target changed from polygon/material')
            targets.append(missing);fronts.append(pair[0])
        region=validate_hidden_attachment_region(self.modules[a]['mask'],targets,c['a']['point'])
        yy,xx=np.nonzero(region);x,y=int(xx.min()),int(yy.min());right,bottom=int(xx.max())+1,int(yy.max())+1
        own=np.zeros((bottom-y,right-x),float)
        material_by_name={p['name']:p for p in self.parts_json['parts']}
        for name in self.modules[a]['pieceNames']:
            self._verified_canvas_alpha(name,material_by_name[name]['canvasLayer']['sha256'])
            pd=self.pieces[name]
            alpha=_render_slot_crop(self.r,pd['slot'],pd['reference'],[x,y,right-x,bottom-y])[...,3]/255.0
            own=alpha+own*(1-alpha)
        missing=region.copy();missing[y:bottom,x:right]&=own<250/255.0
        yy,xx=np.nonzero(missing)
        self.hidden_attachments.append({'id':c['id'],'parent':a,'owner':self.modules[a]['owner'],
            'fronts':list(dict.fromkeys(fronts)),'points':np.column_stack((xx+.5,yy+.5)),
            'registeredPixels':int(region.sum()),'samples':[]})

    def _observe_hidden_attachments(self,t,gain,posed,transforms):
        for entry in self.hidden_attachments:
            world=apply_affine(entry['points'],transforms[entry['owner']]);coverage=np.zeros(len(world),float)
            if len(world):
                x=max(0,int(np.floor(world[:,0].min())));y=max(0,int(np.floor(world[:,1].min())))
                right=min(self.r.width,int(np.ceil(world[:,0].max()))+1)
                bottom=min(self.r.height,int(np.ceil(world[:,1].max()))+1)
                if right>x and bottom>y:
                    xx,yy=np.floor(world[:,0]).astype(int)-x,np.floor(world[:,1]).astype(int)-y
                    valid=(xx>=0)&(yy>=0)&(xx<right-x)&(yy<bottom-y)
                    for name in entry['fronts']:
                        pd=self.pieces[name];r=_render_slot_crop(self.r,pd['slot'],posed[name],[x,y,right-x,bottom-y])
                        alpha=r[yy[valid],xx[valid],3]/255.0
                        coverage[valid]=alpha+coverage[valid]*(1-alpha)
            zero=coverage<=0;low=(coverage>0)&(coverage<250/255.0)
            entry['samples'].append({'time':round(float(t),9),'gain':float(gain),
                'uncoveredPixels':int(zero.sum()),'lowAlphaPixels':int(low.sum()),
                'firstUncoveredCanvasPoint':world[np.flatnonzero(zero)[0]].tolist() if zero.any() else None})

    def _finish_hidden_attachments(self):
        requested=[c for c in self.contract.get('contacts',[]) if c.get('type')=='attachment' and c.get('hiddenBy')]
        if requested and (self.errors or len(self.hidden_attachments)!=len(requested)):
            return result('fail',reason='Canvas attachment evidence validation failed',errors=self.errors)
        checks={};duration=float(self.report.get('duration',8));fps=max(1,int(self.r.fps))
        required={round(float(t),9) for t in np.linspace(0,duration,int(math.ceil(duration*fps))+1)}
        for entry in self.hidden_attachments:
            rows=entry['samples'];dense=required.issubset({r['time'] for r in rows if abs(r['gain']-1)<1e-6})
            zero=max((r['uncoveredPixels'] for r in rows),default=0);low=max((r['lowAlphaPixels'] for r in rows),default=0)
            checks[entry['id']]=result('fail' if zero else 'pending' if low or not dense else 'pass',
                fullCycleSampled=dense,requiredFramePoses=len(required),maxUncoveredPixels=zero,maxLowAlphaPixels=low,
                registeredPixels=entry['registeredPixels'],unfilledReferencePixels=len(entry['points']),
                allowedOccluders=entry['fronts'],samples=rows)
        return result(aggregate(checks) if checks else 'not_applicable',attachments=checks,
            scope='declared missing parent corridor behind named Canvas layers; zero alpha fails, low alpha remains pending; not complete anatomy/visual approval')

    def _prepare_hidden_joint(self,c)->None:
        from src.studio.module_contract import validate_hidden_joint_region
        a,b=c['a']['module'],c['b']['module']
        am,bm=self.modules[a]['declaration'],self.modules[b]['declaration']
        if (c['type']!='joint' or bm.get('parent')!=a
                or any(m.get('shape')!='rigid' or m.get('role') not in BODY_ROLES-{'foot'} for m in (am,bm))):
            raise ValueError('Invalid hidden joint type/parent: '+c['id'])
        refs=c['hiddenBy']
        if not isinstance(refs,list) or not 1<=len(refs)<=8:raise ValueError('Invalid hidden joint references')
        orders={s['name']:i for i,s in enumerate(self.r.sk.slots)}
        targets=[];fronts=[];pairs=set()
        for ref in refs:
            if not isinstance(ref,dict) or set(ref)!={'front','back'} or any(not isinstance(v,str) for v in ref.values()):
                raise ValueError('Invalid hidden joint pair')
            pair=(ref.get('front'),ref.get('back'))
            if pair in pairs:raise ValueError('Duplicate hidden joint reference')
            pairs.add(pair)
            front=self.pieces.get(pair[0]);back=self.pieces.get(pair[1])
            if front is None or back is None or back['module']!=a or orders[pair[0]]<=orders[pair[1]]:
                raise ValueError('Invalid hidden joint layer references')
            fm=self.modules[front['module']]['declaration']
            connected=any(v.get('a',{}).get('module')==a and v.get('b',{}).get('module')==front['module']
                          for v in self.contract.get('contacts',[]))
            if ((front['module']!=b and (fm.get('parent')!=a or not connected))
                    or fm.get('role') in ('weapon','effect_group','pet','prop','foot')):
                raise ValueError('Unrelated foreground cannot authorize hidden joint')
            o=next((v for v in self.contract.get('occlusions',[]) if not v.get('autoSeamUnderlap')
                    and (v.get('front'),v.get('back'))==pair),None)
            if o is None:raise ValueError('Missing hidden joint occlusion')
            _,_,target=read_evidence_mask(o['sourceHiddenTargetMask'],o.get('sourceHiddenTargetMaskSHA256'),
                (self.r.width,self.r.height),root=self.evidence_root,base=self.evidence_root)
            if (target&~front['declaredMask']).any() or (target&back['declaredMask']).any():
                raise ValueError('Hidden joint reference escapes declared front material')
            from src.studio.vision_schema import rasterize_polygons
            normalized=[[[x/self.r.width,y/self.r.height] for x,y in polygon] for polygon in o.get('polygons') or []]
            expected=rasterize_polygons(normalized,self.r.width,self.r.height)&front['declaredMask']&~back['declaredMask']
            if not np.array_equal(target,expected):
                raise ValueError('Hidden joint reference differs from declared source region')
            targets.append(target);fronts.append(pair[0])
        region=validate_hidden_joint_region(self.modules[a]['mask'],targets,c['a']['point'],self.source[...,3])
        yy,xx=np.nonzero(region);x,y=int(xx.min()),int(yy.min());right,bottom=int(xx.max())+1,int(yy.max())+1
        own=np.zeros((bottom-y,right-x),float)
        for name in self.modules[a]['pieceNames']:
            pd=self.pieces[name]
            alpha=_render_slot_crop(self.r,pd['slot'],pd['reference'],[x,y,right-x,bottom-y])[...,3]/255.0
            own=alpha+own*(1-alpha)
        missing=region.copy();missing[y:bottom,x:right]&=own<250/255.0
        yy,xx=np.nonzero(missing)
        self.hidden_joints.append({'id':c['id'],'parent':a,'owner':self.modules[a]['owner'],
                                  'fronts':list(dict.fromkeys(fronts)),
                                  'points':np.column_stack((xx+.5,yy+.5)),
                                  'registeredPixels':int(region.sum()),'samples':[]})

    def _observe_hidden_joints(self,t,gain,posed,transforms)->None:
        for entry in self.hidden_joints:
            world=apply_affine(entry['points'],transforms[entry['owner']])
            coverage=np.zeros(len(world),float)
            if len(world):
                x=max(0,int(np.floor(world[:,0].min())));y=max(0,int(np.floor(world[:,1].min())))
                right=min(self.r.width,int(np.ceil(world[:,0].max()))+1)
                bottom=min(self.r.height,int(np.ceil(world[:,1].max()))+1)
                if right>x and bottom>y:
                    xx,yy=np.floor(world[:,0]).astype(int)-x,np.floor(world[:,1]).astype(int)-y
                    valid=(xx>=0)&(yy>=0)&(xx<right-x)&(yy<bottom-y)
                    for name in entry['fronts']:
                        pd=self.pieces[name]
                        rendered=_render_slot_crop(self.r,pd['slot'],posed[name],[x,y,right-x,bottom-y])
                        alpha=rendered[yy[valid],xx[valid],3]/255.0
                        coverage[valid]=alpha+coverage[valid]*(1-alpha)
            bad=coverage<250/255.0
            entry['samples'].append({'time':round(float(t),9),'gain':float(gain),
                'uncoveredPixels':int(bad.sum()),
                'firstUncoveredCanvasPoint':world[np.flatnonzero(bad)[0]].tolist() if bad.any() else None})

    def audit_hidden_joint_cycle(self)->None:
        """Check every output-frame pose and loop end without rendering whole scenes.

        Existing regular/stress observations still measure rigidity and drift.
        This extra sampling cannot bless anatomy; it proves only named coverage.
        """
        if not (self.hidden_joints or self.hidden_attachments) or self.errors:return
        duration=float(self.report.get('duration',8.0));fps=int(self.r.fps)
        if duration<=0 or not 1<=fps<=120:raise ValueError('Invalid hidden joint cycle sampling')
        count=int(math.ceil(duration*fps))
        for t in np.linspace(0,duration,count+1):
            self.r.pose(float(t))
            entries=self.hidden_joints+self.hidden_attachments
            needed={name for entry in entries for name in entry['fronts']}
            parents={entry['parent'] for entry in entries}
            needed|={name for mid in parents for name in self.modules[mid]['pieceNames']}
            posed={name:self.r.to_canvas(self.r.sk.attachment_vertices(self.pieces[name]['slot'])[0]) for name in needed}
            transforms={self.modules[mid]['owner']:fit_affine(self.modules[mid]['reference'],
                        np.vstack([posed[n] for n in self.modules[mid]['pieceNames']]))[0] for mid in parents}
            self._observe_hidden_joints(float(t),1.0,posed,transforms)
            self._observe_hidden_attachments(float(t),1.0,posed,transforms)

    def _finish_hidden_joints(self)->dict:
        checks={};duration=float(self.report.get('duration',8.0));fps=max(1,int(self.r.fps))
        required={round(float(t),9) for t in np.linspace(0,duration,int(math.ceil(duration*fps))+1)}
        for entry in self.hidden_joints:
            rows=entry['samples'];normal={v['time'] for v in rows if abs(v['gain']-1)<1e-9}
            worst=max((v['uncoveredPixels'] for v in rows),default=0)
            dense=required.issubset(normal)
            checks[entry['id']]=result('fail' if worst else 'pass' if dense else 'pending',
                maxUncoveredPixels=worst,fullCycleSampled=dense,requiredFramePoses=len(required),
                unfilledReferencePixels=len(entry['points']),registeredPixels=entry['registeredPixels'],
                allowedOccluders=entry['fronts'],samples=rows)
        return result(aggregate(checks) if checks else 'not_applicable',joints=checks,
            scope='actual named foreground coverage of declared missing parent material at every output-frame pose plus stress samples; no back fill required while hidden; drift remains mandatory')

    def observe(self,t:float,gain:float,frame:np.ndarray)->None:
        if not self.applicable or self.errors:return
        posed={};transforms={};visible_tracks={};motion_bound=0.0
        for name,p in self.pieces.items():
            world,uv,tris,_=self.r.sk.attachment_vertices(p['slot']);v=self.r.to_canvas(world)
            if uv.shape!=p['uv'].shape or tris.shape!=p['triangles'].shape or not np.array_equal(tris,p['triangles']):
                self.material_changed.add(name)
            else:self.uv_drift=max(self.uv_drift,float(np.abs(uv-p['uv']).max(initial=0)))
            if hashlib.sha256(memoryview(np.ascontiguousarray(self.r.tex[name]))).hexdigest()!=p['textureSHA256']:self.material_changed.add(name)
            posed[name]=v
            motion_bound=max(motion_bound,float(np.linalg.norm(v-p['reference'],axis=1).max(initial=0)))
        for ident,info in self.modules.items():
            decl=info['declaration']
            if decl.get('lockedRoot'):
                reason=locked_root_runtime_error(decl, self.r.sk.by_name)
                if reason:
                    self.locked_root_invalid.add(ident)
                else:
                    rec={'gain':float(gain),
                         'driver':float(self.r.sk.by_name[decl.get('bone')].rotation),
                         'first':float(self.r.sk.by_name[decl.get('softBones')[0]].rotation)}
                    self.locked_root_rotations.setdefault(ident, []).append(rec)
            current=np.vstack([posed[n]for n in info['pieceNames']]);ref=info['reference']
            matrix,residual=fit_affine(ref,current);transforms[info['owner']]=matrix
            hard=rigidity_metrics(ref,current)
            if info['declaration']['shape']=='soft':
                for bn,before in info['boneReferences'].items():
                    actual=self.r.sk.by_name[bn].matrix()[:,:2]@np.linalg.inv(before)
                    singular=np.linalg.svd(actual,compute_uv=False)
                    info['softBoneSingularDeviation']=max(info['softBoneSingularDeviation'],float(np.max(np.abs(singular-1))))
                    den=max(1e-12,float(np.linalg.norm(actual[:,0])*np.linalg.norm(actual[:,1])))
                    info['softBoneShear']=max(info['softBoneShear'],abs(float(actual[:,0]@actual[:,1]))/den)
                if (hard['affineResidualPx']<=max(1e-6,LIMITS['rigid_residual_px_at1080']*self.scale)
                    and (hard['singularDeviation']>LIMITS['rigid_singular_deviation']or hard['shear']>LIMITS['rigid_shear'])):
                    self.uniform_soft_warp.add(ident)
            for k,value in hard.items():
                if k=='determinant':info['rigidMax'][k]=min(info['rigidMax'].get(k,math.inf),value)
                else:info['rigidMax'][k]=max(info['rigidMax'].get(k,0.0),value)
            visible=[];original=[]
            for n in info['pieceNames']:
                pd=self.pieces[n];ids=pd['sampleIndices'];w=pd['sampleWeights'];pts=(posed[n][ids]*w[:,:,None]).sum(axis=1)
                vis=self._actual_visible_samples(n,pts,posed)
                visible.append(pts[vis]);original.append(pd['referenceSamples'][vis])
                sm=soft_deformation_metrics(pd['reference'],posed[n],pd['triangles'])
                for k,value in sm.items():
                    info['softMax'][k]=max(info['softMax'].get(k,0),value)
                    if abs(float(gain)-1.0)<1e-6:
                        info['softMaxNormal'][k]=max(info['softMaxNormal'].get(k,0),value)
            v=np.vstack(visible);r=np.vstack(original);maxmove=float(np.linalg.norm(v-r,axis=1).max(initial=0))
            if abs(gain-1.0)<1e-9:
                info['motionMax']=max(info['motionMax'],maxmove);info['tracks'].append((matrix,v,r))
            visible_tracks[ident]=(v,r)
        hole=self.negative.evaluate(frame[...,3],transforms,motion_bound,posed);hole.update(time=round(t,6),gain=gain);self.hole_samples.append(hole)
        self._observe_remaining_hidden(t,gain,transforms)
        self._observe_hidden_joints(t,gain,posed,transforms)
        self._observe_hidden_attachments(t,gain,posed,transforms)
        for c in self.contacts:
            positions=[];endpoint_matrices=[];unsupported=False
            for key in('a','b'):
                ep=c.get(key)
                if ep is None:continue
                info=self.modules[ep['module']]
                if info['declaration']['shape']=='rigid':
                    matrix=transforms[info['owner']];pos=apply_affine(ep['point'][None],matrix)[0]
                elif ep['indices'][0]>=0:
                    pos=(posed[ep['piece']][ep['indices']]*ep['weights'][:,None]).sum(axis=0)
                    matrix,_=fit_affine(self.pieces[ep['piece']]['reference'][ep['indices']],posed[ep['piece']][ep['indices']])
                else:unsupported=True;break
                positions.append(pos)
                endpoint_matrices.append(matrix)
            if unsupported:
                self.pending.append('Contact outside supported soft material: '+c['id']);continue
            if len(positions)==1:
                positions.append(c['world']);endpoint_matrices.append(np.eye(3))
            if c['type']=='grip':
                # These are setup-to-current material transforms, including ancestors
                # and skinning. A common rigid motion cancels; rotation about the
                # unchanged grip point does not. Joints and soft attachments are
                # positional constraints, so they must not inherit this fixed pose.
                a,b=(matrix[:2,:2]for matrix in endpoint_matrices)
                try:
                    relative=np.linalg.solve(a,b)
                    valid=bool(np.isfinite(relative).all()and np.linalg.det(relative)>0)
                    u,_,vt=np.linalg.svd(relative);rotation=u@vt
                    deviation=float(np.linalg.norm(relative-np.eye(2),ord=2))
                    angle=abs(math.degrees(math.atan2(rotation[1,0],rotation[0,0])))
                    sample={'valid':valid,'linearDeviation':deviation,'rotationDegrees':angle}
                except np.linalg.LinAlgError:
                    sample={'valid':False,'linearDeviation':None,'rotationDegrees':None}
                self.grip_pose_samples[c['id']].append(sample)
            drift=float(np.linalg.norm(positions[0]-positions[1]));self.contact_samples[c['id']].append(drift)
            if c['type']in('foot','foot_anchor')and 'world'in c:
                self.contact_samples[c['id']][-1]=max(drift,float(np.linalg.norm(positions[0]-c['world'])))
            samples=(apply_affine(c['coveragePoints'],endpoint_matrices[0])+apply_affine(c['coveragePoints'],endpoint_matrices[1]))/2
            if len(samples):
                xi=np.floor(samples[:,0]).astype(int);yi=np.floor(samples[:,1]).astype(int)
                valid=(xi>=0)&(yi>=0)&(xi<self.r.width)&(yi<self.r.height)
                empty=np.ones(len(samples),bool);empty[valid]=frame[yi[valid],xi[valid],3]<128
                self.contact_gap_samples[c['id']].append(int(empty.sum()))
        self.samples.append({'time':round(t,6),'gain':gain})

    def _actual_visible_samples(self,name:str,points:np.ndarray,posed:Dict[str,np.ndarray])->np.ndarray:
        """Use material alpha and later rendered attachments, not merely animated bones."""
        p=self.pieces[name];uv=(p['uv'][p['sampleIndices']]*p['sampleWeights'][:,:,None]).sum(axis=1)
        tex=self.r.tex[name];th,tw=tex.shape[:2]
        opacity=sample_texture(tex,uv[:,0]*tw-.5,uv[:,1]*th-.5,self.r.kernel)[:,3].astype(float)
        after=False
        for slot in self.r.sk.slots:
            if slot['name']==name:after=True;continue
            if not after or slot['name'] not in posed:continue
            other=self.pieces[slot['name']];vertices=posed[slot['name']]
            lo,hi=vertices.min(axis=0),vertices.max(axis=0)
            todo=np.flatnonzero((opacity>=.5)&np.all(points>=lo,axis=1)&np.all(points<=hi,axis=1))
            if not len(todo):continue
            inds,weights=_point_bind(points[todo],vertices,other['triangles']);good=inds[:,0]>=0
            if not good.any():continue
            uv2=(other['uv'][inds[good]]*weights[good,:,None]).sum(axis=1)
            tex2=self.r.tex[slot['name']];h,w=tex2.shape[:2]
            alpha=sample_texture(tex2,uv2[:,0]*w-.5,uv2[:,1]*h-.5,self.r.kernel)[:,3]
            opacity[todo[good]]*=1-np.clip(alpha,0,255)/255
        return (opacity>=.5)&(points[:,0]>=0)&(points[:,1]>=0)&(points[:,0]<self.r.width)&(points[:,1]<self.r.height)

    def finish(self)->Dict[str,Any]:
        if not self.applicable:
            return {'rulesVersion':RULES_VERSION,'contractApplicable':False,
                    'inputOwnership':result('not_applicable',reason='Legacy output has no module contract; no module-quality assertion'),
                    'bindingMotion':result('not_applicable'), 'machineQC':result('not_applicable'),
                    'visual':{'status':'pending','required':True,'reasons':['Legacy output is not approved under the module contract']},
                    'overall':{'status':'pending','exportAllowed':False}}
        if self.errors:
            self.checks_input['contract']=result('fail',errors=self.errors)
        else:self.checks_input['contract']=result('pass')
        hard_bad=[];soft_topo_bad=[];soft_numeric_advisory=[];soft_driver_bad=[];still=[];moving_static=[];module_metrics={}
        for ident,info in self.modules.items():
            hard=info['rigidMax'];soft=info['softMax'];soft_n=info.get('softMaxNormal')or {};m=info['declaration']
            if not hard:
                self.pending.append('No geometry samples: '+ident);continue
            if m['shape']=='rigid':
                if (hard['singularDeviation']>LIMITS['rigid_singular_deviation']or hard['distanceRelativeError']>LIMITS['rigid_distance_relative_error']
                    or hard['shear']>LIMITS['rigid_shear']or hard['affineResidualPx']>max(1e-6,LIMITS['rigid_residual_px_at1080']*self.scale)or hard['determinant']<=0):hard_bad.append(ident)
            else:
                if soft.get('foldedTriangles',0) or soft.get('degenerateTriangles',0):
                    soft_topo_bad.append(ident)
                if (soft_n.get('maxSingularDeviation',0)>LIMITS['soft_singular_deviation']or soft_n.get('maxAreaDeviation',0)>LIMITS['soft_area_deviation']
                    or soft_n.get('maxAdjacentRotationDegrees',0)>LIMITS['soft_adjacent_rotation_degrees']
                    or soft_n.get('rotationSpreadDegrees',0)>LIMITS['soft_rotation_spread_degrees']):
                    soft_numeric_advisory.append(ident)
            if m['shape']=='soft'and (info['softBoneSingularDeviation']>LIMITS['rigid_singular_deviation']or info['softBoneShear']>LIMITS['rigid_shear']):soft_driver_bad.append(ident)
            if m['motion']=='required'and info['motionMax']<LIMITS['required_motion_px_at1080']*self.scale:still.append(ident)
            if m['motion']=='static'and info['motionMax']>LIMITS['static_motion_px_at1080']*self.scale:moving_static.append(ident)
            module_metrics[ident]={'shape':m['shape'],'motion':m['motion'],'actualVisibleDisplacementPx':round(info['motionMax'],6),
                                   'rigidity':hard,'softDeformation':soft,'softDeformationNormalPlayback':soft_n,'pieces':info['pieceNames']}
            module_metrics[ident]['softInfluenceWorldSingularDeviation']=info['softBoneSingularDeviation']
            module_metrics[ident]['softInfluenceWorldShear']=info['softBoneShear']
            if m.get('lockedRoot'):
                module_metrics[ident]['lockedRoot']=True
        self.checks_binding['rigidity']=result('fail'if hard_bad else'pass',modules=hard_bad,thresholds={k:v for k,v in LIMITS.items()if k.startswith('rigid')})
        self.checks_binding['softDeformation']=result(
            'fail' if soft_topo_bad else 'pass',
            modules=soft_topo_bad,
            advisoryModules=soft_numeric_advisory,
            advisory=True,
            advisoryScope='normal-playback gain=1.0 numeric strain/curvature; topology remains a hard gate',
            thresholds={k:v for k,v in LIMITS.items()if k.startswith('soft')},
            hardConstraints=['foldedTriangles==0','degenerateTriangles==0'])
        self.checks_binding['moduleInfluences']=result('fail'if self.foreign_influences else'pass',foreignInfluences=self.foreign_influences)
        self.checks_binding['softDriverRigidity']=result('fail'if soft_driver_bad or self.uniform_soft_warp else'pass',
            nonRigidWorldInfluenceModules=soft_driver_bad,uniformScaleOrShearModules=sorted(self.uniform_soft_warp),
            scope='soft bend is not permission for whole-module scale/shear; actual world bone matrices and uniform affine geometry checked')
        self.checks_binding['visibleMotion']=result('fail'if still or moving_static else'pass',stillRequiredModules=still,movedStaticModules=moving_static,
                                                  minimumPx=LIMITS['required_motion_px_at1080']*self.scale)
        moving=[info for info in self.modules.values()if info['declaration']['motion']=='required'
                and info['declaration']['shape']=='rigid'
                and info['declaration'].get('role')in BODY_ROLES and info['tracks']]
        relative=0.0
        if len(moving)>=2:
            master=moving[0]
            for info in moving[1:]:
                for (base,_,_),(matrix,v,r) in zip(master['tracks'],info['tracks']):
                    delta=np.linalg.norm(v-apply_affine(r,base),axis=1)
                    relative=max(relative,float(delta.max(initial=0)))
            self.checks_binding['relativeMotion']=result('pass'if relative>=LIMITS['relative_motion_px_at1080']*self.scale else'fail',
                maxRelativeVisibleDisplacementPx=relative,minimumPx=LIMITS['relative_motion_px_at1080']*self.scale,
                test='subtract actual common whole-picture affine motion; rotations are not false relative motion')
        else:self.checks_binding['relativeMotion']=result('pending',reason='At least two required rigid anatomical modules are needed to prove coordinated relative motion')
        contact_drift={k:max(v)if v else None for k,v in self.contact_samples.items()};tol=max(.25,LIMITS['contact_drift_px_at1080']*self.scale)
        bad_contacts=[k for k,v in contact_drift.items()if v is not None and v>tol]
        self.checks_binding['contacts']=result('fail'if bad_contacts else('pending'if any(v is None for v in contact_drift.values())else'pass'),
            maxDriftPx=contact_drift,failedContacts=bad_contacts,thresholdPx=tol)
        grip_metrics={k:{'maxRelativeLinearDeviation':max((s['linearDeviation']for s in v if s['linearDeviation']is not None),default=None),
                         'maxRelativeRotationDegrees':max((s['rotationDegrees']for s in v if s['rotationDegrees']is not None),default=None)}
                      for k,v in self.grip_pose_samples.items()}
        invalid_grips=[k for k,v in self.grip_pose_samples.items()if any(not s['valid']for s in v)]
        failed_grips=[k for k,m in grip_metrics.items()if k in invalid_grips or
                      (m['maxRelativeLinearDeviation']is not None and m['maxRelativeLinearDeviation']>LIMITS['grip_relative_linear_deviation'])]
        self.checks_binding['gripRelativePose']=result('fail'if failed_grips else
            ('pending'if any(not v for v in self.grip_pose_samples.values())else('pass'if grip_metrics else'not_applicable')),
            contacts=grip_metrics,failedContacts=failed_grips,invalidMaterialTransforms=invalid_grips,
            thresholdLinearDeviation=LIMITS['grip_relative_linear_deviation'],
            scope='fixed grip setup-relative material pose; point drift checked separately; joint/attachment rotation allowed')
        locked_fail=list(self.locked_root_invalid); locked_metrics={}
        contract_modules=(self.contract or {}).get('modules') or []
        for m in contract_modules:
            if not m.get('lockedRoot'):
                continue
            ident=m.get('id')
            if m.get('motion')!='required':
                locked_fail.append(ident)
            if locked_root_runtime_error(m, self.r.sk.by_name):
                locked_fail.append(ident)
        for ident, samples in self.locked_root_rotations.items():
            by_gain={}
            for sample in samples:
                by_gain.setdefault(round(sample['gain'], 3), []).append(sample)
            normal_ptp=stress_ptp=0.0
            for g, rows in by_gain.items():
                ptp=max((max(r[k] for r in rows)-min(r[k] for r in rows) for k in ('driver','first')), default=0.0)
                if abs(g-1.0)<1e-6:
                    normal_ptp=max(normal_ptp, ptp)
                    if ptp>0.6: locked_fail.append(ident)
                elif g>1.0:
                    stress_ptp=max(stress_ptp, ptp)
                    if ptp>0.9: locked_fail.append(ident)
            locked_metrics[ident]={'normalPtpDegrees':normal_ptp,'stressPtpDegrees':stress_ptp}
        has_locked=any(m.get('lockedRoot') for m in ((self.contract or {}).get('modules') or []))
        self.checks_binding['lockedRoot']=result(
            'fail' if locked_fail else ('pass' if has_locked else 'not_applicable'),
            modules=sorted(set(locked_fail)), metrics=locked_metrics,
            normalLimitDegrees=0.6, stressLimitDegrees=0.9)
        holes_status=aggregate({str(i):s for i,s in enumerate(self.hole_samples)})if self.hole_samples else'pending'
        self.checks_machine['negativeSpace']=result('pass',advisory=True,measuredStatus=holes_status,samples=self.hole_samples,
            newHolePixelsMax=max((s['newHolePixels']for s in self.hole_samples),default=0),
            ambiguousPixelsMax=max((s['ambiguousPixels']for s in self.hole_samples),default=0),
            unresolvedOriginalHolesMax=max((s.get('unresolvedOriginalHoles')or 0 for s in self.hole_samples),default=0),
            scope='advisory: original cross-module voids and dynamic enclosed air are recorded, not hard-fail; rest_diff still guards visible RGBA')
        gaps={k:max(v)if v else 0 for k,v in self.contact_gap_samples.items()}
        self.checks_machine['contactSweepCoverage']=result('fail'if any(gaps.values())else'pass',
            uncoveredReferenceContactPixelsMax=gaps,scope='declared contact interior patches; no claim to cover every unclassified exterior seam')
        self.checks_machine['remainingHiddenCoverage']=self._finish_remaining_hidden()
        self.checks_machine['hiddenJointCoverage']=self._finish_hidden_joints()
        self.checks_machine['hiddenAttachmentCoverage']=self._finish_hidden_attachments()
        self.checks_machine['materialStability']=result('fail'if self.material_changed or self.uv_drift>1e-9 else'pass',
            mutatedTextureOrTopologyPieces=sorted(self.material_changed),maxUVDrift=self.uv_drift,
            scope='actual texture bytes and material UVs stay stable; geometry deformation is measured separately')
        if self.pending:self.checks_machine['unsupported']=result('pending',reasons=sorted(set(self.pending)))
        q={'rulesVersion':RULES_VERSION,'contractApplicable':True,'thresholdCalibration':'engineering guards; not calibrated for every portrait',
           'bodyHeightPx':self.body_height_px,'thresholdScale':self.scale,
           'inputOwnership':result(aggregate(self.checks_input),checks=self.checks_input,reasons=['Declared coverage is not semantic completeness; visual review must inspect actual source and edges']),
           'bindingMotion':result(aggregate(self.checks_binding),checks=self.checks_binding,modules=module_metrics),
           'machineQC':result(aggregate(self.checks_machine),checks=self.checks_machine,samples=self.samples),
           'visual':{'status':'pending','required':True,'reasons':['Semantic completeness, foreign fringes, original matte contamination, and normal full-loop appearance require independent review']+self.hidden_risks}}
        return recompute_quality({'quality':q})['quality']
