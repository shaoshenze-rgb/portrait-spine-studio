import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

import {
  bezierValue,
  sampleTimeline,
  deformAt,
  Spine42Skeleton,
} from '../../src/player/spine42-runtime.js';
import { computeConvexHull } from '../../src/player/canvas-renderer.js';
import { normalizeHealthReport } from '../../src/player/player-widget.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const PLAYER_DIR = path.resolve(__dirname, '../../src/player');

function assertNear(actual, expected, tol = 1e-9, msg = '') {
  const diff = Math.abs(actual - expected);
  assert.ok(
    diff <= tol,
    `${msg ? msg + ': ' : ''}expected ${expected} +- ${tol}, got ${actual} (diff=${diff})`
  );
}

const tests = [];
function test(name, fn) {
  tests.push({ name, fn });
}

// ---------------------------------------------------------------------------
// 1. Bezier curve solver accuracy & boundary clamping
// ---------------------------------------------------------------------------
test('Bezier curve solver accuracy & boundary clamping', () => {
  // Linear-equivalent cubic bezier
  const t0 = 1.0, v0 = -10.0, t1 = 5.0, v1 = 30.0;
  const linCurve = [t0 + (t1 - t0) / 3, v0 + (v1 - v0) / 3, t0 + (2 * (t1 - t0)) / 3, v0 + (2 * (v1 - v0)) / 3];
  for (const frac of [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]) {
    const t = t0 + (t1 - t0) * frac;
    const expected = v0 + (v1 - v0) * frac;
    assertNear(bezierValue(t, t0, v0, t1, v1, linCurve), expected, 1e-9, `linear frac=${frac}`);
  }

  // Symmetric S-curve (ease-in-out) midpoint
  const sCurve = [t0 + (t1 - t0) * 0.5, v0, t0 + (t1 - t0) * 0.5, v1];
  assertNear(bezierValue(3.0, t0, v0, t1, v1, sCurve), 10.0, 1e-9, 'S-curve midpoint');

  // Asymmetric cubic bezier verified against exact parametric evaluation at known s
  const asym = [1.8, -2.0, 4.1, 25.0];
  for (const s of [0.2, 0.45, 0.7, 0.85]) {
    const u = 1 - s;
    const xs = u * u * u * t0 + 3 * u * u * s * asym[0] + 3 * u * s * s * asym[2] + s * s * s * t1;
    const ys = u * u * u * v0 + 3 * u * u * s * asym[1] + 3 * u * s * s * asym[3] + s * s * s * v1;
    assertNear(bezierValue(xs, t0, v0, t1, v1, asym), ys, 1e-9, `asymmetric s=${s}`);
  }

  // Boundary clamping & stepped timeline
  const keys = [
    { time: 1.0, value: 5.0, curve: 'stepped' },
    { time: 3.0, value: 15.0 },
    { time: 5.0, value: 25.0 },
  ];
  assert.deepEqual(sampleTimeline(keys, -2.0, ['value'], [0]), [5.0], 'clamp before first key');
  assert.deepEqual(sampleTimeline(keys, 1.0, ['value'], [0]), [5.0], 'at first key');
  assert.deepEqual(sampleTimeline(keys, 2.999, ['value'], [0]), [5.0], 'stepped holds previous value');
  assert.deepEqual(sampleTimeline(keys, 4.0, ['value'], [0]), [20.0], 'linear midpoint');
  assert.deepEqual(sampleTimeline(keys, 99.0, ['value'], [0]), [25.0], 'clamp after last key');
});

// ---------------------------------------------------------------------------
// 2. Hierarchical FK world transforms & bone timelines
// ---------------------------------------------------------------------------
test('Hierarchical FK world transforms & bone timelines', () => {
  const skJson = {
    skeleton: { spine: '4.2.43', x: -200, y: 0, width: 400, height: 600 },
    bones: [
      { name: 'root', x: 10, y: 20, rotation: 30, scaleX: 1.5, scaleY: 1.2 },
      { name: 'torso', parent: 'root', x: 40, y: 0, rotation: 45, length: 80, shearX: 5, shearY: -3 },
      { name: 'head', parent: 'torso', x: 80, y: 0, rotation: -15, length: 50 },
    ],
    slots: [],
    skins: [{ name: 'default', attachments: {} }],
    animations: {
      test_anim: {
        bones: {
          root: {
            translate: [{ time: 0, x: 0, y: 0 }, { time: 2, x: 20, y: -10 }],
            scale: [{ time: 0, x: 1, y: 1 }, { time: 2, x: 2, y: 0.5 }],
          },
          torso: {
            rotate: [{ time: 0, value: 0 }, { time: 2, value: 30 }],
            translatex: [{ time: 0, value: 0 }, { time: 2, value: 10 }],
            translatey: [{ time: 0, value: 0 }, { time: 2, value: -6 }],
            shear: [{ time: 0, x: 0, y: 0 }, { time: 2, x: 4, y: 6 }],
          },
        },
      },
    },
  };

  const sk = new Spine42Skeleton(skJson);
  const root = sk.byName.get('root');
  const torso = sk.byName.get('torso');
  const head = sk.byName.get('head');

  // Verify analytical setup world transform for root -> torso
  const r30 = (30 * Math.PI) / 180;
  assertNear(root.worldX, 10, 1e-12);
  assertNear(root.worldY, 20, 1e-12);
  assertNear(root.a, Math.cos(r30) * 1.5, 1e-12);
  assertNear(root.c, Math.sin(r30) * 1.5, 1e-12);
  assertNear(torso.worldX, root.a * 40 + root.worldX, 1e-12);
  assertNear(torso.worldY, root.c * 40 + root.worldY, 1e-12);

  // Verify head origin == torso.localToWorld(80, 0)
  const [hx, hy] = torso.localToWorld(80, 0);
  assertNear(head.worldX, hx, 1e-12);
  assertNear(head.worldY, hy, 1e-12);

  // Verify animation timelines at midpoint t = 1.0
  sk.apply('test_anim', 1.0);
  assertNear(root.x, 10 + 10, 1e-12);
  assertNear(root.y, 20 - 5, 1e-12);
  assertNear(root.scaleX, 1.5 * 1.5, 1e-12);
  assertNear(root.scaleY, 1.2 * 0.75, 1e-12);
  assertNear(torso.rotation, 45 + 15, 1e-12);
  assertNear(torso.x, 40 + 5, 1e-12);
  assertNear(torso.y, 0 - 3, 1e-12);
  assertNear(torso.shearX, 5 + 2, 1e-12);
  assertNear(torso.shearY, -3 + 3, 1e-12);
});

// ---------------------------------------------------------------------------
// 3. 1-bone and 2-bone IK accuracy & bendPositive flip
// ---------------------------------------------------------------------------
test('1-bone and 2-bone IK accuracy and bendPositive direction', () => {
  // 1-bone IK under rotated/scaled ancestor
  const sk1 = new Spine42Skeleton({
    skeleton: { spine: '4.2.43', width: 400, height: 400 },
    bones: [
      { name: 'root', x: 10, y: 20, rotation: 25, scaleX: 1.2, scaleY: 0.8 },
      { name: 'aim', parent: 'root', x: 30, y: 15, length: 60 },
      { name: 'target', x: 140, y: 185 },
    ],
    ik: [{ name: 'aim_ik', order: 0, bones: ['aim'], target: 'target' }],
    slots: [],
    skins: [{ name: 'default', attachments: {} }],
    animations: {},
  });

  const aim = sk1.byName.get('aim');
  const tgt1 = sk1.byName.get('target');
  const [tipX, tipY] = aim.localToWorld(aim.length, 0);
  // Vector from aim origin to tip should be collinear with vector from aim origin to target
  const v1x = tipX - aim.worldX, v1y = tipY - aim.worldY;
  const v2x = tgt1.worldX - aim.worldX, v2y = tgt1.worldY - aim.worldY;
  const cross = (v1x * v2y - v1y * v2x) / (Math.hypot(v1x, v1y) * Math.hypot(v2x, v2y));
  const dot = (v1x * v2x + v1y * v2y) / (Math.hypot(v1x, v1y) * Math.hypot(v2x, v2y));
  assertNear(cross, 0.0, 1e-9, '1-bone IK collinearity');
  assertNear(dot, 1.0, 1e-9, '1-bone IK positive direction');

  // 2-bone IK with bendPositive = true vs false under rotated/scaled ancestor
  const make2Bone = (bendPositive, tx, ty) =>
    new Spine42Skeleton({
      skeleton: { spine: '4.2.43', width: 500, height: 500 },
      bones: [
        { name: 'root', x: 25, y: 40, rotation: -18, scaleX: 1.15, scaleY: 1.15 },
        { name: 'upper', parent: 'root', x: 20, y: 10, length: 100 },
        { name: 'lower', parent: 'upper', x: 100, y: 0, length: 85 },
        { name: 'ik_target', x: tx, y: ty },
      ],
      ik: [{ name: 'leg_ik', order: 0, bones: ['upper', 'lower'], target: 'ik_target', bendPositive }],
      slots: [],
      skins: [{ name: 'default', attachments: {} }],
      animations: {},
    });

  const testTargets = [
    [130, 120],
    [-40, 160],
    [175, -15],
  ];

  for (const [tx, ty] of testTargets) {
    const skPos = make2Bone(true, tx, ty);
    const skNeg = make2Bone(false, tx, ty);

    for (const [label, sk] of [['bendPositive=true', skPos], ['bendPositive=false', skNeg]]) {
      const lower = sk.byName.get('lower');
      const [endX, endY] = lower.localToWorld(lower.length, 0);
      const err = Math.hypot(endX - tx, endY - ty);
      assert.ok(err < 1e-4, `${label} target (${tx},${ty}) error ${err} >= 1e-4`);
    }

    // Verify knee/elbow joint sits on opposite sides of the upper-origin -> target line
    const uPos = skPos.byName.get('upper');
    const jPos = skPos.byName.get('lower');
    const jNeg = skNeg.byName.get('lower');
    const side = (jx, jy) =>
      (tx - uPos.worldX) * (jy - uPos.worldY) - (ty - uPos.worldY) * (jx - uPos.worldX);
    const sPos = side(jPos.worldX, jPos.worldY);
    const sNeg = side(jNeg.worldX, jNeg.worldY);
    assert.ok(sPos * sNeg < 0, `bendPositive should flip joint side: sPos=${sPos}, sNeg=${sNeg}`);
    assertNear(sPos, -sNeg, 1e-4, 'symmetric bend flip magnitude');
  }

  // Unreachable target stretches straight toward target without NaN
  const skFar = make2Bone(true, 1000, 1000);
  const lowerFar = skFar.byName.get('lower');
  const [fx, fy] = lowerFar.localToWorld(lowerFar.length, 0);
  assert.ok(Number.isFinite(fx) && Number.isFinite(fy), 'unreachable IK produces finite coords');
});

// ---------------------------------------------------------------------------
// 4. Weighted mesh vertex skinning, region attachments & deform timelines
// ---------------------------------------------------------------------------
test('Weighted mesh vertex skinning, region attachments, and deform', () => {
  const skJson = {
    skeleton: { spine: '4.2.43', x: -100, y: 0, width: 200, height: 300 },
    bones: [
      { name: 'root', x: 0, y: 0 },
      { name: 'b0', parent: 'root', x: 10, y: 20, rotation: 0 },
      { name: 'b1', parent: 'b0', x: 50, y: 0, rotation: 90 },
    ],
    slots: [
      { name: 'reg_slot', bone: 'b0', attachment: 'box' },
      { name: 'mesh_slot', bone: 'b0', attachment: 'ribbon' },
      { name: 'deform_slot', bone: 'b0', attachment: 'strip' },
    ],
    skins: [
      {
        name: 'default',
        attachments: {
          reg_slot: {
            box: { type: 'region', x: 0, y: 0, width: 20, height: 10 },
          },
          mesh_slot: {
            ribbon: {
              type: 'mesh',
              uvs: [0, 0, 1, 0, 1, 1],
              triangles: [0, 1, 2],
              // 3 weighted vertices:
              // v0: 100% b0 (0, 0)
              // v1: 70% b0 (10, 5) + 30% b1 (20, 0)
              // v2: 25% b0 (0, 10) + 75% b1 (0, 10)
              vertices: [
                1, 1, 0, 0, 1.0,
                2, 1, 10, 5, 0.7, 2, 20, 0, 0.3,
                2, 1, 0, 10, 0.25, 2, 0, 10, 0.75,
              ],
            },
          },
          deform_slot: {
            strip: {
              type: 'mesh',
              uvs: [0, 0, 1, 0, 0, 1],
              triangles: [0, 1, 2],
              vertices: [0, 0, 10, 0, 0, 10],
            },
          },
        },
      },
    ],
    animations: {
      bend: {
        attachments: {
          default: {
            deform_slot: {
              strip: {
                deform: [
                  { time: 0, offset: 0, vertices: [0, 0, 0, 0, 0, 0] },
                  { time: 2, offset: 2, vertices: [4, 6] },
                ],
              },
            },
          },
        },
      },
    },
  };

  const sk = new Spine42Skeleton(skJson);
  const b0 = sk.byName.get('b0');
  const b1 = sk.byName.get('b1');

  // Weighted mesh verification
  const meshGeom = sk.computeWorldGeometry(sk.slots[1]);
  assert.equal(meshGeom.vertexCount, 3);
  assert.equal(meshGeom.triangleCount, 1);

  // v0: b0(0,0) = (10, 20)
  assertNear(meshGeom.worldVertices[0], 10, 1e-9);
  assertNear(meshGeom.worldVertices[1], 20, 1e-9);
  // Canvas space: originX=100, originY=300 -> (110, 280)
  assertNear(meshGeom.vertices[0], 110, 1e-9);
  assertNear(meshGeom.vertices[1], 280, 1e-9);

  // v1: 0.7 * b0(10, 5) + 0.3 * b1(20, 0)
  const [p1aX, p1aY] = b0.localToWorld(10, 5);
  const [p1bX, p1bY] = b1.localToWorld(20, 0);
  assertNear(meshGeom.worldVertices[2], 0.7 * p1aX + 0.3 * p1bX, 1e-9);
  assertNear(meshGeom.worldVertices[3], 0.7 * p1aY + 0.3 * p1bY, 1e-9);

  // v2: 0.25 * b0(0, 10) + 0.75 * b1(0, 10)
  const [p2aX, p2aY] = b0.localToWorld(0, 10);
  const [p2bX, p2bY] = b1.localToWorld(0, 10);
  assertNear(meshGeom.worldVertices[4], 0.25 * p2aX + 0.75 * p2bX, 1e-9);
  assertNear(meshGeom.worldVertices[5], 0.25 * p2aY + 0.75 * p2bY, 1e-9);

  // Region quad verification
  const regGeom = sk.computeWorldGeometry(sk.slots[0]);
  assert.equal(regGeom.vertexCount, 4);
  assert.equal(regGeom.triangleCount, 2);
  // BL corner (-10, -5) in b0=(10,20) -> world (0, 15) -> canvas (100, 285)
  assertNear(regGeom.worldVertices[0], 0, 1e-9);
  assertNear(regGeom.worldVertices[1], 15, 1e-9);
  assertNear(regGeom.vertices[0], 100, 1e-9);
  assertNear(regGeom.vertices[1], 285, 1e-9);

  // Deform timeline at midpoint t=1.0 -> offset 2 gets (+2, +3) on vertex 1 (10, 0) -> (12, 3)
  sk.apply('bend', 1.0);
  const dfmGeom = sk.computeWorldGeometry(sk.slots[2]);
  assertNear(dfmGeom.worldVertices[2], 10 + 12, 1e-9);
  assertNear(dfmGeom.worldVertices[3], 20 + 3, 1e-9);

  // Convex hull helper sanity
  const hull = computeConvexHull(regGeom.vertices);
  assert.equal(hull.length, 4);
});

// ---------------------------------------------------------------------------
// 5. 0s and 8s seamless loop zero-at-start identity (exact Setup Pose match)
// ---------------------------------------------------------------------------
test('0s and 8s seamless loop matches Setup Pose identically', () => {
  const idleRig = {
    skeleton: { spine: '4.2.43', x: -502, y: 0, width: 1004, height: 1542, fps: 30, images: './images/' },
    bones: [
      { name: 'root' },
      { name: 'hips', parent: 'root', x: 0, y: 720, length: 120 },
      { name: 'chest', parent: 'hips', x: 0, y: 120, length: 160 },
      { name: 'head', parent: 'chest', x: 0, y: 160, length: 130 },
      { name: 'thigh_L', parent: 'hips', x: -45, y: -10, rotation: -95, length: 300 },
      { name: 'shin_L', parent: 'thigh_L', x: 300, y: 0, rotation: 8, length: 300 },
      { name: 'foot_L_ik', parent: 'root', x: -65, y: 155 },
    ],
    ik: [
      { name: 'leg_L', order: 0, bones: ['thigh_L', 'shin_L'], target: 'foot_L_ik', bendPositive: false },
    ],
    slots: [
      { name: 'torso', bone: 'chest', attachment: 'torso' },
      { name: 'head_slot', bone: 'head', attachment: 'head' },
    ],
    skins: [
      {
        name: 'default',
        attachments: {
          torso: {
            torso: {
              type: 'mesh',
              uvs: [0, 0, 1, 0, 1, 1, 0, 1],
              triangles: [0, 1, 2, 0, 2, 3],
              vertices: [
                2, 1, -60, 0, 0.5, 2, -60, -120, 0.5,
                2, 1, 60, 0, 0.5, 2, 60, -120, 0.5,
                1, 2, 60, 140, 1.0,
                1, 2, -60, 140, 1.0,
              ],
            },
          },
          head_slot: {
            head: { type: 'region', x: 0, y: 60, width: 140, height: 160 },
          },
        },
      },
    ],
    animations: {
      idle: {
        bones: {
          hips: {
            translatey: [
              { time: 0.0, value: 0.0, curve: [1.5, 0.0, 2.5, 12.0] },
              { time: 4.0, value: 12.0, curve: [5.5, 12.0, 6.5, 0.0] },
              { time: 8.0, value: 0.0 },
            ],
          },
          chest: {
            rotate: [
              { time: 0.0, value: 0.0, curve: [2.0, 0.0, 2.0, 2.8] },
              { time: 4.0, value: 2.8, curve: [6.0, 2.8, 6.0, 0.0] },
              { time: 8.0, value: 0.0 },
            ],
          },
          head: {
            rotate: [
              { time: 0.0, value: 0.0, curve: [2.0, 0.0, 2.0, -3.5] },
              { time: 4.0, value: -3.5, curve: [6.0, -3.5, 6.0, 0.0] },
              { time: 8.0, value: 0.0 },
            ],
          },
        },
      },
    },
  };

  const sk = new Spine42Skeleton(idleRig);
  assert.equal(sk.getAnimationDuration('idle'), 8.0);

  // Capture Setup Pose geometry
  sk.setupPose();
  sk.updateWorld();
  const setupVerts = sk.slots.map((s) => Array.from(sk.computeWorldGeometry(s).vertices));

  // Verify t = 0.0 and t = 8.0 match Setup Pose identically
  for (const t of [0.0, 8.0]) {
    sk.apply('idle', t);
    let maxErr = 0.0;
    for (let s = 0; s < sk.slots.length; s++) {
      const verts = sk.computeWorldGeometry(sk.slots[s]).vertices;
      for (let i = 0; i < verts.length; i++) {
        maxErr = Math.max(maxErr, Math.abs(verts[i] - setupVerts[s][i]));
      }
    }
    assert.ok(maxErr < 1e-12, `t=${t}s should match Setup Pose identically (maxErr=${maxErr})`);
  }

  // Verify t = 4.0 actually moves and IK foot stays anchored
  sk.apply('idle', 4.0);
  let midDelta = 0.0;
  for (let s = 0; s < sk.slots.length; s++) {
    const verts = sk.computeWorldGeometry(sk.slots[s]).vertices;
    for (let i = 0; i < verts.length; i++) {
      midDelta = Math.max(midDelta, Math.abs(verts[i] - setupVerts[s][i]));
    }
  }
  assert.ok(midDelta > 5.0, `t=4.0s should exhibit non-trivial idle motion (midDelta=${midDelta})`);

  const shin = sk.byName.get('shin_L');
  const footTgt = sk.byName.get('foot_L_ik');
  const [footX, footY] = shin.localToWorld(shin.length, 0);
  assert.ok(Math.hypot(footX - footTgt.worldX, footY - footTgt.worldY) < 1e-4, 'IK foot stays anchored at t=4.0s');

  // Verify wireframe helper outputs
  const segs = sk.getBoneWorldSegments();
  assert.equal(segs.length, idleRig.bones.length);
  const iks = sk.getIkTargets();
  assert.equal(iks.length, 1);
  assert.equal(iks[0].name, 'leg_L');
});

// ---------------------------------------------------------------------------
// 6. Health report normalization sanity
// ---------------------------------------------------------------------------
test('Health report normalization covers 4 hard gates and 4 soft hints', () => {
  const report = normalizeHealthReport({
    hard_gates: {
      rest_diff: { pass: true, value: 0.4, threshold: 2.0, unit: 'px' },
      setup_identity: { pass: true, value: 0.0 },
      no_holes: { pass: true, count: 0 },
      anchor_stability: { pass: true, max_px: 0.12, threshold: 0.5, unit: 'px' },
    },
    soft_hints: {
      head: { value: 6.4, status: 'pass' },
      chest: { value: 4.1, status: 'pass' },
      hips: { value: 3.2, status: 'pass' },
      hair_tip: { value: 14.8, status: 'info' },
    },
  });
  assert.equal(report.loaded, true);
  assert.equal(report.overallStatus, 'pass');
  assert.equal(report.gates.length, 4);
  assert.equal(report.hints.length, 4);
  assert.deepEqual(
    report.gates.map((g) => g.key),
    ['rest_diff', 'setup_identity', 'no_holes', 'anchor_stability']
  );
  assert.deepEqual(
    report.hints.map((h) => h.key),
    ['head', 'chest', 'hips', 'hair_tip']
  );

  // Canonical T2 schema (docs/contracts/spine42-schema.md)
  const canonical = normalizeHealthReport({
    version: '1.0',
    status: 'pass',
    hardGates: {
      rest_diff: {
        passed: true,
        threshold: { maxRGBA: 2 },
        measured: { maxRGBA: 0, meanAbsDiff: 0.0, pixelsOver2: 0 },
      },
      setup_identity: {
        passed: true,
        threshold: { frame0VsSetup: 0, frameLoopEndVsSetup: 0, setupVsPartsMaxRGBA: 2 },
        measured: { frame0VsSetup: 0, frameLoopEndVsSetup: 0, setupVsPartsMaxRGBA: 1 },
      },
      no_holes: {
        passed: true,
        threshold: { enclosedHolesMax: 0 },
        measured: { loopEnclosedHolesMax: 0, stressEnclosedHolesMax: 0, enclosedHolesMax: 0 },
      },
      anchor_stability: {
        passed: true,
        threshold: { maxDriftPx: 1.5 },
        measured: { maxDriftPx: 0.02, perConstraint: { left_arm_ik: 0.02 } },
      },
    },
    softHints: {
      preset: 'agile_full',
      gain: 1.0,
      bodyHeightPx: 1200,
      landmarks: {
        head: { bone: 'head', ptpPx: 12.4, refRangePx: [4.0, 28.0], status: 'ok' },
        chest: { bone: 'chest', ptpPx: 2.1, refRangePx: [3.0, 20.0], status: 'warn_low' },
        hips: { bone: 'hips', ptpPx: 25.5, refRangePx: [2.0, 22.0], status: 'warn_high' },
        hair_tip: { bone: null, ptpPx: null, refRangePx: [8.0, 65.0], status: 'not_present' },
      },
    },
  });
  assert.equal(canonical.loaded, true);
  assert.equal(canonical.overallStatus, 'pass');
  assert.deepEqual(
    canonical.gates.map((g) => [g.key, g.status, g.value]),
    [
      ['rest_diff', 'pass', 0],
      ['setup_identity', 'pass', 1],
      ['no_holes', 'pass', 0],
      ['anchor_stability', 'pass', 0.02],
    ]
  );
  assert.ok(canonical.gates.every((g) => g.valueText !== '--'));
  assert.deepEqual(
    canonical.hints.map((h) => [h.key, h.status, h.value]),
    [
      ['head', 'pass', 12.4],
      ['chest', 'warn', 2.1],
      ['hips', 'warn', 25.5],
      ['hair_tip', 'na', null],
    ]
  );
  assert.equal(canonical.hints[0].valueText, '12.40 px');
  assert.equal(canonical.hints[3].valueText, '--');
});

// ---------------------------------------------------------------------------
// 7. Syntax check (`node --check`) on all JS files in src/player/
// ---------------------------------------------------------------------------
test('node --check passes on all JS files in src/player/', () => {
  const jsFiles = readdirSync(PLAYER_DIR).filter((f) => f.endsWith('.js') || f.endsWith('.mjs'));
  assert.ok(jsFiles.length >= 3, `Expected at least 3 JS files in ${PLAYER_DIR}, found ${jsFiles.length}`);
  for (const f of jsFiles) {
    const abs = path.join(PLAYER_DIR, f);
    execFileSync(process.execPath, ['--check', abs], { stdio: 'pipe' });
  }
});

// ---------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------
let passed = 0;
let failed = 0;

for (const t of tests) {
  try {
    t.fn();
    passed++;
    console.log(`PASS: ${t.name}`);
  } catch (err) {
    failed++;
    console.error(`FAIL: ${t.name}`);
    console.error(err);
  }
}

console.log(`\nSummary: ${passed}/${tests.length} passed, ${failed} failed.`);
if (failed > 0) {
  process.exit(1);
}
