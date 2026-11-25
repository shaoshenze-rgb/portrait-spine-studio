/**
 * Zero-dependency Spine 4.2.43 JSON runtime (pure ES module, browser + Node.js).
 *
 * Mathematically aligned with `spine_runtime.py`:
 * - Bones with normal inheritance (x, y, rotation, scaleX/Y, shearX/Y) & hierarchical updateWorld()
 * - 1-bone and 2-bone IK solver in parent local space with bendPositive & subtree updates
 * - Bone timelines (rotate, translate, translatex, translatey, scale, shear) with stepped, linear,
 *   and exact cubic Bezier curve evaluation via 60-step bisection solving x(s) = t
 * - Region attachments and unweighted / weighted mesh attachments + deform timelines
 * - Canvas-space geometry, bone wireframe segments, and IK target helpers
 */

const DEG_TO_RAD = Math.PI / 180;
const RAD_TO_DEG = 180 / Math.PI;
const EPS = 1e-12;
const MAX_UNIFORM_IK_STRETCH = 1.06;

/**
 * Evaluate a Spine 4.2 cubic Bezier segment from (t0, v0) to (t1, v1) with absolute control points
 * c = [cx1, cy1, cx2, cy2] by solving x(s) = t via bisection.
 *
 * @param {number} t
 * @param {number} t0
 * @param {number} v0
 * @param {number} t1
 * @param {number} v1
 * @param {number[]} c
 * @returns {number}
 */
export function bezierValue(t, t0, v0, t1, v1, c) {
  if (t <= t0) return v0;
  if (t >= t1) return v1;
  const cx1 = c[0];
  const cy1 = c[1];
  const cx2 = c[2];
  const cy2 = c[3];

  let lo = 0.0;
  let hi = 1.0;
  for (let iter = 0; iter < 60; iter++) {
    const s = 0.5 * (lo + hi);
    const u = 1.0 - s;
    const x =
      u * u * u * t0 +
      3 * u * u * s * cx1 +
      3 * u * s * s * cx2 +
      s * s * s * t1;
    if (x < t) {
      lo = s;
    } else {
      hi = s;
    }
  }
  const s = 0.5 * (lo + hi);
  const u = 1.0 - s;
  return (
    u * u * u * v0 +
    3 * u * u * s * cy1 +
    3 * u * s * s * cy2 +
    s * s * s * v1
  );
}

/**
 * Evaluate a multi-field Spine 4.2 keyframe list at time t.
 *
 * @param {Array<object>} keys
 * @param {number} t
 * @param {string[]} fields
 * @param {number[]} defaults
 * @returns {number[]}
 */
export function sampleTimeline(keys, t, fields, defaults) {
  if (!keys || keys.length === 0) {
    return defaults.slice();
  }
  const count = keys.length;
  const firstTime = keys[0].time ?? 0.0;
  if (t <= firstTime || count === 1) {
    return fields.map((f, idx) => (keys[0][f] !== undefined ? keys[0][f] : defaults[idx]));
  }
  const lastKey = keys[count - 1];
  const lastTime = lastKey.time ?? 0.0;
  if (t >= lastTime) {
    return fields.map((f, idx) => (lastKey[f] !== undefined ? lastKey[f] : defaults[idx]));
  }

  let i = 0;
  for (let j = 0; j < count; j++) {
    if ((keys[j].time ?? 0.0) <= t) {
      i = j;
    } else {
      break;
    }
  }

  const k0 = keys[i];
  const k1 = keys[i + 1];
  const t0 = k0.time ?? 0.0;
  const t1 = k1.time ?? 0.0;
  const curve = k0.curve;

  if (curve === 'stepped') {
    return fields.map((f, idx) => (k0[f] !== undefined ? k0[f] : defaults[idx]));
  }

  const out = new Array(fields.length);
  const isBezier = Array.isArray(curve);
  const dt = t1 - t0;

  for (let n = 0; n < fields.length; n++) {
    const f = fields[n];
    const d = defaults[n];
    const v0 = k0[f] !== undefined ? k0[f] : d;
    const v1 = k1[f] !== undefined ? k1[f] : d;
    if (isBezier && curve.length >= 4 * n + 4) {
      out[n] = bezierValue(t, t0, v0, t1, v1, [
        curve[4 * n],
        curve[4 * n + 1],
        curve[4 * n + 2],
        curve[4 * n + 3],
      ]);
    } else {
      out[n] = dt > 0 ? v0 + ((v1 - v0) * (t - t0)) / dt : v0;
    }
  }
  return out;
}

/**
 * Evaluate a Spine 4.2 attachment deform timeline at time t.
 *
 * @param {Array<object>} keys
 * @param {number} t
 * @returns {Float64Array}
 */
export function deformAt(keys, t) {
  if (!keys || keys.length === 0) {
    return new Float64Array(0);
  }
  const count = keys.length;
  let i = 0;
  let s = 0.0;
  const firstTime = keys[0].time ?? 0.0;
  const lastTime = keys[count - 1].time ?? 0.0;

  if (t <= firstTime || count === 1) {
    i = 0;
    s = 0.0;
  } else if (t >= lastTime) {
    i = count - 1;
    s = 0.0;
  } else {
    for (let j = 0; j < count; j++) {
      if ((keys[j].time ?? 0.0) <= t) {
        i = j;
      } else {
        break;
      }
    }
    const k0 = keys[i];
    const k1 = keys[i + 1];
    const t0 = k0.time ?? 0.0;
    const t1 = k1.time ?? 0.0;
    const curve = k0.curve;
    if (curve === 'stepped') {
      s = 0.0;
    } else if (Array.isArray(curve) && curve.length >= 4) {
      s = bezierValue(t, t0, 0.0, t1, 1.0, curve);
    } else {
      s = t1 > t0 ? (t - t0) / (t1 - t0) : 0.0;
    }
  }

  let n = 0;
  for (let j = 0; j < count; j++) {
    const off = keys[j].offset ?? 0;
    const vs = keys[j].vertices ?? [];
    if (off + vs.length > n) n = off + vs.length;
  }

  const out = new Float64Array(n);
  const kA = keys[i];
  const offA = kA.offset ?? 0;
  const vsA = kA.vertices ?? [];
  for (let j = 0; j < vsA.length; j++) {
    out[offA + j] = vsA[j];
  }

  if (s !== 0.0 && i + 1 < count) {
    const denseB = new Float64Array(n);
    const kB = keys[i + 1];
    const offB = kB.offset ?? 0;
    const vsB = kB.vertices ?? [];
    for (let j = 0; j < vsB.length; j++) {
      denseB[offB + j] = vsB[j];
    }
    for (let j = 0; j < n; j++) {
      out[j] = out[j] + (denseB[j] - out[j]) * s;
    }
  }
  return out;
}

function atan2Deg(y, x) {
  return Math.atan2(y, x) * RAD_TO_DEG;
}

function isClose(a, b, relTol = 1e-9, absTol = 1e-12) {
  return Math.abs(a - b) <= Math.max(relTol * Math.max(Math.abs(a), Math.abs(b)), absTol);
}

/**
 * Spine 4.2 Bone with normal inheritance and 2x3 world affine transform.
 */
export class Bone {
  /**
   * @param {object} data
   * @param {Bone|null} parent
   */
  constructor(data, parent = null) {
    this.name = data.name;
    this.parent = parent;
    this.data = data;
    this.length = data.length ?? 0.0;

    this.x = 0.0;
    this.y = 0.0;
    this.rotation = 0.0;
    this.scaleX = 1.0;
    this.scaleY = 1.0;
    this.shearX = 0.0;
    this.shearY = 0.0;

    this.a = 1.0;
    this.b = 0.0;
    this.c = 0.0;
    this.d = 1.0;
    this.worldX = 0.0;
    this.worldY = 0.0;

    this.setup();
  }

  setup() {
    const d = this.data;
    this.x = d.x ?? 0.0;
    this.y = d.y ?? 0.0;
    this.rotation = d.rotation ?? 0.0;
    this.scaleX = d.scaleX ?? 1.0;
    this.scaleY = d.scaleY ?? 1.0;
    this.shearX = d.shearX ?? 0.0;
    this.shearY = d.shearY ?? 0.0;
  }

  /**
   * Recompute world matrix (a, b, c, d, worldX, worldY) from local properties and parent world matrix.
   *
   * @param {number} [rootX=0]
   * @param {number} [rootY=0]
   */
  updateWorld(rootX = 0.0, rootY = 0.0) {
    const rx = (this.rotation + this.shearX) * DEG_TO_RAD;
    const ry = (this.rotation + 90.0 + this.shearY) * DEG_TO_RAD;
    const la = Math.cos(rx) * this.scaleX;
    const lb = Math.cos(ry) * this.scaleY;
    const lc = Math.sin(rx) * this.scaleX;
    const ld = Math.sin(ry) * this.scaleY;

    const p = this.parent;
    if (p === null) {
      this.a = la;
      this.b = lb;
      this.c = lc;
      this.d = ld;
      this.worldX = this.x + rootX;
      this.worldY = this.y + rootY;
    } else {
      const inheritMode = this.data.inherit ?? this.data.transform ?? 'normal';
      if (inheritMode !== 'normal') {
        throw new Error(`Unsupported bone inherit mode "${inheritMode}" on bone "${this.name}"`);
      }
      this.a = p.a * la + p.b * lc;
      this.b = p.a * lb + p.b * ld;
      this.c = p.c * la + p.d * lc;
      this.d = p.c * lb + p.d * ld;
      this.worldX = p.a * this.x + p.b * this.y + p.worldX;
      this.worldY = p.c * this.x + p.d * this.y + p.worldY;
    }
  }

  /**
   * Transform a local bone-space point into Spine world space (y-up).
   *
   * @param {number} lx
   * @param {number} ly
   * @returns {[number, number]}
   */
  localToWorld(lx, ly) {
    return [
      this.a * lx + this.b * ly + this.worldX,
      this.c * lx + this.d * ly + this.worldY,
    ];
  }

  /**
   * Transform a Spine world-space point into this bone's local space.
   *
   * @param {number} wx
   * @param {number} wy
   * @returns {[number, number]}
   */
  worldToLocal(wx, wy) {
    const det = this.a * this.d - this.b * this.c;
    if (Math.abs(det) < EPS) {
      throw new Error(`Singular bone world transform on "${this.name}"`);
    }
    const dx = wx - this.worldX;
    const dy = wy - this.worldY;
    const invDet = 1.0 / det;
    return [
      (this.d * dx - this.b * dy) * invDet,
      (-this.c * dx + this.a * dy) * invDet,
    ];
  }
}

/**
 * Spine 4.2.43 Skeleton evaluator.
 */
export class Spine42Skeleton {
  /**
   * @param {object|string} jsonOrString
   */
  constructor(jsonOrString) {
    const d = typeof jsonOrString === 'string' ? JSON.parse(jsonOrString) : jsonOrString;
    if (!d || typeof d !== 'object') {
      throw new Error('Spine42Skeleton requires a valid Spine 4.2 JSON object');
    }
    this.json = d;
    this.skeleton = d.skeleton ?? {};

    /** @type {Bone[]} */
    this.bones = [];
    /** @type {Map<string, Bone>} */
    this.byName = new Map();
    /** @type {Map<string, number>} */
    this.boneIndex = new Map();
    /** @type {Map<string, Bone[]>} */
    this.children = new Map();

    const boneDefs = d.bones ?? [];
    for (let i = 0; i < boneDefs.length; i++) {
      const bd = boneDefs[i];
      const parent = bd.parent ? (this.byName.get(bd.parent) ?? null) : null;
      const bone = new Bone(bd, parent);
      this.bones.push(bone);
      this.byName.set(bone.name, bone);
      this.boneIndex.set(bone.name, i);
      this.children.set(bone.name, []);
    }
    for (const b of this.bones) {
      if (b.parent !== null) {
        this.children.get(b.parent.name).push(b);
      }
    }

    this.slots = d.slots ?? [];
    const skins = d.skins ?? [];
    const defaultSkin = skins.find((s) => s.name === 'default') ?? skins[0] ?? { attachments: {} };
    this.attachments = defaultSkin.attachments ?? {};

    this.ik = (d.ik ?? []).slice().sort((a, b) => (a.order ?? 0) - (b.order ?? 0));
    /** @type {Map<string, Float64Array>} */
    this.deform = new Map();

    this.setupPose();
    this.updateWorld();
  }

  /**
   * Reset all bones to setup pose and clear active deform offsets.
   */
  setupPose() {
    for (let i = 0; i < this.bones.length; i++) {
      this.bones[i].setup();
    }
    this.deform.clear();
  }

  /**
   * List available animation names.
   * @returns {string[]}
   */
  getAnimationNames() {
    return Object.keys(this.json.animations ?? {});
  }

  /**
   * Compute the duration (max keyframe time) of an animation.
   * @param {string} animName
   * @returns {number}
   */
  getAnimationDuration(animName) {
    const anim = this.json.animations?.[animName];
    if (!anim) return 0.0;
    let maxT = 0.0;
    const bones = anim.bones ?? {};
    for (const bname of Object.keys(bones)) {
      const tls = bones[bname];
      for (const kind of Object.keys(tls)) {
        const keys = tls[kind];
        if (Array.isArray(keys) && keys.length > 0) {
          const lastT = keys[keys.length - 1].time ?? 0.0;
          if (lastT > maxT) maxT = lastT;
        }
      }
    }
    const atts = anim.attachments ?? {};
    for (const skinName of Object.keys(atts)) {
      const slots = atts[skinName];
      for (const slotName of Object.keys(slots)) {
        const slotAtts = slots[slotName];
        for (const attName of Object.keys(slotAtts)) {
          const keys = slotAtts[attName]?.deform;
          if (Array.isArray(keys) && keys.length > 0) {
            const lastT = keys[keys.length - 1].time ?? 0.0;
            if (lastT > maxT) maxT = lastT;
          }
        }
      }
    }
    return maxT;
  }

  /**
   * Apply setup pose + animation at time t (MixBlend.setup, alpha = 1) and update world transforms + IK.
   *
   * @param {string} animName
   * @param {number} t
   * @param {{ loop?: boolean, duration?: number|null }} [options]
   */
  apply(animName, t, options = {}) {
    this.setupPose();
    const anim = this.json.animations?.[animName];
    if (!anim) {
      throw new Error(`Animation "${animName}" not found`);
    }

    let evalT = t;
    if (options.loop) {
      const dur = options.duration ?? this.getAnimationDuration(animName);
      if (dur > 0) {
        evalT = ((t % dur) + dur) % dur;
      }
    }

    const animBones = anim.bones ?? {};
    for (const bname of Object.keys(animBones)) {
      const b = this.byName.get(bname);
      if (!b) continue;
      const sd = b.data;
      const tls = animBones[bname];
      for (const kind of Object.keys(tls)) {
        const keys = tls[kind];
        if (kind === 'rotate') {
          const [val] = sampleTimeline(keys, evalT, ['value'], [0.0]);
          b.rotation = (sd.rotation ?? 0.0) + val;
        } else if (kind === 'translate') {
          const [x, y] = sampleTimeline(keys, evalT, ['x', 'y'], [0.0, 0.0]);
          b.x = (sd.x ?? 0.0) + x;
          b.y = (sd.y ?? 0.0) + y;
        } else if (kind === 'translatex') {
          const [val] = sampleTimeline(keys, evalT, ['value'], [0.0]);
          b.x = (sd.x ?? 0.0) + val;
        } else if (kind === 'translatey') {
          const [val] = sampleTimeline(keys, evalT, ['value'], [0.0]);
          b.y = (sd.y ?? 0.0) + val;
        } else if (kind === 'scale') {
          const [x, y] = sampleTimeline(keys, evalT, ['x', 'y'], [1.0, 1.0]);
          b.scaleX = (sd.scaleX ?? 1.0) * x;
          b.scaleY = (sd.scaleY ?? 1.0) * y;
        } else if (kind === 'shear') {
          const [x, y] = sampleTimeline(keys, evalT, ['x', 'y'], [0.0, 0.0]);
          b.shearX = (sd.shearX ?? 0.0) + x;
          b.shearY = (sd.shearY ?? 0.0) + y;
        } else {
          throw new Error(`Unsupported bone timeline "${kind}"`);
        }
      }
    }

    const animAtts = anim.attachments ?? {};
    for (const skinName of Object.keys(animAtts)) {
      const slots = animAtts[skinName];
      for (const slotName of Object.keys(slots)) {
        const atts = slots[slotName];
        for (const attName of Object.keys(atts)) {
          const keys = atts[attName]?.deform;
          if (keys && keys.length > 0) {
            this.deform.set(`${slotName}\0${attName}`, deformAt(keys, evalT));
          }
        }
      }
    }

    this.updateWorld();
  }

  /**
   * Update all bone world transforms in hierarchy order, then solve IK constraints in order.
   */
  updateWorld() {
    for (let i = 0; i < this.bones.length; i++) {
      this.bones[i].updateWorld();
    }
    for (let i = 0; i < this.ik.length; i++) {
      this._applyIK(this.ik[i]);
    }
  }

  /**
   * Recursively update world transforms for a bone and its descendants.
   * @param {Bone} bone
   */
  _updateSubtree(bone) {
    bone.updateWorld();
    const kids = this.children.get(bone.name);
    if (!kids) return;
    for (let i = 0; i < kids.length; i++) {
      this._updateSubtree(kids[i]);
    }
  }

  /**
   * Solve a 1-bone or 2-bone IK constraint in the chain parent's local space.
   * Matches `spine_runtime.py._apply_ik`.
   *
   * @param {object} c
   */
  _applyIK(c) {
    const names = c.bones ?? [];
    const tgt = this.byName.get(c.target);
    if (!tgt) {
      throw new Error(`IK target bone "${c.target}" not found in "${c.name}"`);
    }
    const tx = tgt.worldX;
    const ty = tgt.worldY;

    const stretch = Boolean(c.stretch);
    if ((c.mix ?? 1.0) !== 1.0 || c.softness || c.compress || (stretch && (names.length !== 2 || c.uniform !== true))) {
      throw new Error(`Unsupported IK options in "${c.name}"`);
    }
    if (names.length !== 1 && names.length !== 2) {
      throw new Error(`IK requires one or two bones in "${c.name}"`);
    }

    const parentSpace = (b, px, py) => {
      if (b.parent === null) {
        return [px, py];
      }
      try {
        return b.parent.worldToLocal(px, py);
      } catch (_err) {
        throw new Error(`Singular IK ancestor transform in "${c.name}"`);
      }
    };

    if (names.length === 1) {
      const b = this.byName.get(names[0]);
      if (!b) {
        throw new Error(`IK bone "${names[0]}" not found in "${c.name}"`);
      }
      if (Math.abs(b.scaleX) < EPS) {
        throw new Error(`Zero IK bone scale in "${c.name}"`);
      }
      const [ptx, pty] = parentSpace(b, tx, ty);
      const dx = ptx - b.x;
      const dy = pty - b.y;
      if (Math.hypot(dx, dy) > EPS) {
        b.rotation = atan2Deg(dy, dx) - b.shearX - (b.scaleX < 0 ? 180.0 : 0.0);
      }
      this._updateSubtree(b);
      return;
    }

    const p = this.byName.get(names[0]);
    const ch = this.byName.get(names[1]);
    if (!p || !ch) {
      throw new Error(`IK bones not found in "${c.name}"`);
    }
    if (ch.parent !== p) {
      throw new Error(`IK bones must form a parent-child chain in "${c.name}"`);
    }
    for (const b of [p, ch]) {
      if (
        b.scaleX <= 0 ||
        !isClose(b.scaleX, b.scaleY) ||
        Math.abs(b.shearX) > 1e-9 ||
        Math.abs(b.shearY) > 1e-9
      ) {
        throw new Error(`Unsupported IK chain scale/shear on "${b.name}" in "${c.name}"`);
      }
    }

    const [ptx, pty] = parentSpace(p, tx, ty);
    const dx = ptx - p.x;
    const dy = pty - p.y;
    let l1 = Math.hypot(ch.x, ch.y) * p.scaleX;
    let l2 = (ch.data.length ?? 0.0) * p.scaleX * ch.scaleX;
    if (l1 < EPS || l2 < EPS) {
      throw new Error(`IK chain lengths must be positive in "${c.name}"`);
    }

    const d = Math.hypot(dx, dy);
    if (stretch) {
      if (d < Math.abs(l1 - l2) - 1e-7) throw new Error('IK target inside minimum reach in ' + c.name);
      const ratio = Math.max(1.0, d / (l1 + l2));
      if (ratio > MAX_UNIFORM_IK_STRETCH + 1e-9) throw new Error('IK stretch ' + ratio.toFixed(6) + ' exceeds ' + MAX_UNIFORM_IK_STRETCH.toFixed(2) + ' limit in ' + c.name + '; preserve joints and review motion');
      if (ratio > 1.0) {
        p.scaleX *= ratio;
        p.scaleY *= ratio;
        l1 *= ratio;
        l2 *= ratio;
      }
    }
    const cosa = stretch && Math.abs(d - (l1 + l2)) <= 1e-9
      ? 1.0 : (l1 * l1 + d * d - l2 * l2) / (2.0 * l1 * Math.max(d, 1e-9));
    const clampedCosa = Math.min(1.0, Math.max(-1.0, cosa));
    const alpha = Math.acos(clampedCosa) * RAD_TO_DEG;
    const bend = (c.bendPositive ?? true) ? 1.0 : -1.0;
    const offset = atan2Deg(ch.y, ch.x);
    const direction = d > EPS ? atan2Deg(dy, dx) : p.rotation + offset;

    p.rotation = direction - bend * alpha - offset;
    p.updateWorld();
    ch.updateWorld();

    const [ctx, cty] = parentSpace(ch, tx, ty);
    const cdx = ctx - ch.x;
    const cdy = cty - ch.y;
    if (Math.hypot(cdx, cdy) > EPS) {
      ch.rotation = atan2Deg(cdy, cdx);
    }
    this._updateSubtree(p);
  }

  /**
   * Resolve canvas coordinate mapping from skeleton header metadata.
   * Spine y-up -> Canvas y-down:
   *   canvasX = worldX + originX
   *   canvasY = originY - worldY
   *
   * @param {{ width?: number, height?: number, originX?: number, originY?: number }} [override]
   * @returns {{ width: number, height: number, originX: number, originY: number }}
   */
  getCanvasSpec(override = null) {
    const sk = this.skeleton;
    const width = override?.width ?? sk.width ?? 1004;
    const height = override?.height ?? sk.height ?? 1542;
    const originX =
      override?.originX ?? (sk.x !== undefined ? -sk.x : width * 0.5);
    const originY =
      override?.originY ?? ((sk.y ?? 0.0) + height);
    return { width, height, originX, originY };
  }

  /**
   * Convert Spine world-space coordinates (y-up) to canvas-space coordinates (y-down).
   *
   * @param {number} wx
   * @param {number} wy
   * @param {{ width: number, height: number, originX: number, originY: number }} [spec]
   * @returns {[number, number]}
   */
  worldToCanvas(wx, wy, spec = null) {
    const s = spec ?? this.getCanvasSpec();
    return [wx + s.originX, s.originY - wy];
  }

  /**
   * Compute Spine world-space (y-up) vertices, UVs, and triangle indices for a slot's active attachment.
   *
   * @param {object} slot
   * @returns {{
   *   world: Float64Array,
   *   uvs: Float64Array,
   *   triangles: Uint32Array,
   *   vertexCount: number,
   *   triangleCount: number,
   *   attachment: object,
   *   attachmentName: string
   * } | null}
   */
  attachmentVertices(slot) {
    const attName = slot.attachment;
    if (!attName) return null;
    const att = this.attachments?.[slot.name]?.[attName];
    if (!att) return null;
    const bone = this.byName.get(slot.bone);
    if (!bone) {
      throw new Error(`Slot "${slot.name}" references unknown bone "${slot.bone}"`);
    }

    const type = att.type ?? 'region';
    if (type === 'region') {
      const hw = (att.width ?? 0) * 0.5;
      const hh = (att.height ?? 0) * 0.5;
      const sx = att.scaleX ?? 1.0;
      const sy = att.scaleY ?? 1.0;
      const rad = (att.rotation ?? 0.0) * DEG_TO_RAD;
      const cosR = Math.cos(rad);
      const sinR = Math.sin(rad);
      const ax = att.x ?? 0.0;
      const ay = att.y ?? 0.0;

      // Corners BL, TL, TR, BR in attachment space (y-up), matching spine_runtime.py
      const localCorners = [
        [-hw * sx, -hh * sy],
        [-hw * sx, hh * sy],
        [hw * sx, hh * sy],
        [hw * sx, -hh * sy],
      ];

      const world = new Float64Array(8);
      for (let k = 0; k < 4; k++) {
        const cx = localCorners[k][0];
        const cy = localCorners[k][1];
        // loc @ [[c, s], [-s, c]] + [ax, ay]
        const lx = cx * cosR - cy * sinR + ax;
        const ly = cx * sinR + cy * cosR + ay;
        world[2 * k] = bone.a * lx + bone.b * ly + bone.worldX;
        world[2 * k + 1] = bone.c * lx + bone.d * ly + bone.worldY;
      }

      const uvs = new Float64Array([0, 1, 0, 0, 1, 0, 1, 1]);
      const triangles = new Uint32Array([0, 1, 2, 0, 2, 3]);
      return {
        world,
        uvs,
        triangles,
        vertexCount: 4,
        triangleCount: 2,
        attachment: att,
        attachmentName: attName,
      };
    }

    if (type === 'mesh') {
      const rawUvs = att.uvs ?? [];
      const n = rawUvs.length >> 1;
      const uvs = Float64Array.from(rawUvs);
      const v = att.vertices ?? [];
      const dfm = this.deform.get(`${slot.name}\0${attName}`) ?? null;
      const world = new Float64Array(2 * n);

      if (v.length === 2 * n) {
        // Unweighted mesh in slot bone's local space
        for (let k = 0; k < n; k++) {
          let lx = v[2 * k];
          let ly = v[2 * k + 1];
          if (dfm !== null) {
            lx += dfm[2 * k] ?? 0.0;
            ly += dfm[2 * k + 1] ?? 0.0;
          }
          world[2 * k] = bone.a * lx + bone.b * ly + bone.worldX;
          world[2 * k + 1] = bone.c * lx + bone.d * ly + bone.worldY;
        }
      } else {
        // Weighted mesh
        let idx = 0;
        let f = 0;
        for (let k = 0; k < n; k++) {
          const cnt = v[idx++] | 0;
          let wx = 0.0;
          let wy = 0.0;
          for (let j = 0; j < cnt; j++) {
            const bi = v[idx++] | 0;
            let bx = v[idx++];
            let by = v[idx++];
            const wt = v[idx++];
            if (dfm !== null) {
              bx += dfm[f] ?? 0.0;
              by += dfm[f + 1] ?? 0.0;
            }
            f += 2;
            const mb = this.bones[bi];
            wx += wt * (mb.a * bx + mb.b * by + mb.worldX);
            wy += wt * (mb.c * bx + mb.d * by + mb.worldY);
          }
          world[2 * k] = wx;
          world[2 * k + 1] = wy;
        }
      }

      const triangles = Uint32Array.from(att.triangles ?? []);
      return {
        world,
        uvs,
        triangles,
        vertexCount: n,
        triangleCount: (triangles.length / 3) | 0,
        attachment: att,
        attachmentName: attName,
      };
    }

    throw new Error(`Unsupported attachment type "${type}"`);
  }

  /**
   * Compute canvas-space geometry (vertices, UVs, triangles, bounds) for a slot.
   *
   * @param {object} slot
   * @param {{ width: number, height: number, originX: number, originY: number }} [canvasSpec]
   */
  computeWorldGeometry(slot, canvasSpec = null) {
    const geom = this.attachmentVertices(slot);
    if (!geom) return null;
    const spec = canvasSpec ?? this.getCanvasSpec();
    const n = geom.vertexCount;
    const vertices = new Float64Array(2 * n);

    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;

    for (let k = 0; k < n; k++) {
      const cx = geom.world[2 * k] + spec.originX;
      const cy = spec.originY - geom.world[2 * k + 1];
      vertices[2 * k] = cx;
      vertices[2 * k + 1] = cy;
      if (cx < minX) minX = cx;
      if (cx > maxX) maxX = cx;
      if (cy < minY) minY = cy;
      if (cy > maxY) maxY = cy;
    }

    return {
      slotName: slot.name,
      boneName: slot.bone,
      attachmentName: geom.attachmentName,
      attachment: geom.attachment,
      vertices,
      worldVertices: geom.world,
      uvs: geom.uvs,
      triangles: geom.triangles,
      vertexCount: n,
      triangleCount: geom.triangleCount,
      bounds: { minX, minY, maxX, maxY },
    };
  }

  /**
   * Return canvas-space bone segments for wireframe / rig overlay rendering.
   *
   * @param {{ width: number, height: number, originX: number, originY: number }} [canvasSpec]
   */
  getBoneWorldSegments(canvasSpec = null) {
    const spec = canvasSpec ?? this.getCanvasSpec();
    const ikTargetSet = new Set(this.ik.map((c) => c.target));
    const ikBoneSet = new Set();
    for (const c of this.ik) {
      for (const bname of c.bones ?? []) {
        ikBoneSet.add(bname);
      }
    }

    return this.bones.map((bone) => {
      const [x1, y1] = this.worldToCanvas(bone.worldX, bone.worldY, spec);
      let tipWorldX = bone.worldX;
      let tipWorldY = bone.worldY;
      if (bone.length > 0) {
        [tipWorldX, tipWorldY] = bone.localToWorld(bone.length, 0);
      }
      const [x2, y2] = this.worldToCanvas(tipWorldX, tipWorldY, spec);
      let parentX = null;
      let parentY = null;
      if (bone.parent !== null) {
        [parentX, parentY] = this.worldToCanvas(bone.parent.worldX, bone.parent.worldY, spec);
      }
      return {
        name: bone.name,
        parent: bone.parent ? bone.parent.name : null,
        length: bone.length,
        worldX: bone.worldX,
        worldY: bone.worldY,
        tipWorldX,
        tipWorldY,
        x1,
        y1,
        x2,
        y2,
        parentX,
        parentY,
        isIkTarget: ikTargetSet.has(bone.name),
        isIkBone: ikBoneSet.has(bone.name),
      };
    });
  }

  /**
   * Return canvas-space IK target descriptors for wireframe overlay pins.
   *
   * @param {{ width: number, height: number, originX: number, originY: number }} [canvasSpec]
   */
  getIkTargets(canvasSpec = null) {
    const spec = canvasSpec ?? this.getCanvasSpec();
    const out = [];
    for (const c of this.ik) {
      const tgt = this.byName.get(c.target);
      if (!tgt) continue;
      const [x, y] = this.worldToCanvas(tgt.worldX, tgt.worldY, spec);
      out.push({
        name: c.name,
        targetBone: c.target,
        bones: (c.bones ?? []).slice(),
        bendPositive: c.bendPositive ?? true,
        order: c.order ?? 0,
        worldX: tgt.worldX,
        worldY: tgt.worldY,
        x,
        y,
      });
    }
    return out;
  }
}

export const Skeleton = Spine42Skeleton;
