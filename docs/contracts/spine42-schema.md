# Spine 4.2.43 & Pipeline Data Contracts

Version: `1.0.0`  
Target Spine Runtime: `4.2.43`  
Coordinate Convention:
- **Canvas space**: origin `(0, 0)` at top-left, `+x` right, `+y` down, size `[W, H]`.
- **Spine world space**: origin `(OX, OY)` (default `[W / 2, H]`, bottom-center floor line), `+x` right, `+y` up:
  - `X_spine = x_canvas - OX`
  - `Y_spine = OY - y_canvas`

---

## 1. `<character>.spine.json` (Spine 4.2.43)

```json
{
  "skeleton": {
    "hash": "22-char-sha1-hex",
    "spine": "4.2.43",
    "x": -406.0,
    "y": 0.0,
    "width": 812,
    "height": 1446,
    "fps": 30,
    "images": "./images/"
  },
  "bones": [
    {
      "name": "root",
      "x": 0.0,
      "y": 0.0
    },
    {
      "name": "hips",
      "parent": "root",
      "x": -84.0,
      "y": 704.0,
      "rotation": 90.0,
      "length": 52.0
    }
  ],
  "ik": [
    {
      "name": "left_arm_ik",
      "order": 0,
      "bones": ["shoulder_L", "elbow_L"],
      "target": "wrist_target_L",
      "mix": 1.0,
      "bendPositive": true
    }
  ],
  "slots": [
    {
      "name": "back_hair",
      "bone": "neck",
      "attachment": "back_hair"
    }
  ],
  "skins": [
    {
      "name": "default",
      "attachments": {
        "head": {
          "head": {
            "x": 0.0,
            "y": 40.0,
            "rotation": -90.0,
            "width": 208,
            "height": 228
          }
        },
        "back_hair": {
          "back_hair": {
            "type": "mesh",
            "uvs": [0.0, 0.0, 1.0, 0.0],
            "triangles": [0, 1, 2],
            "vertices": [2, 3, 12.5, -4.2, 0.6, 4, 1.0, 2.0, 0.4],
            "hull": 24,
            "width": 520,
            "height": 680
          }
        }
      }
    }
  ],
  "animations": {
    "idle": {
      "bones": {
        "spine": {
          "rotate": [
            {
              "value": 0.0,
              "curve": [0.04167, 0.01, 0.08333, 0.04]
            },
            {
              "time": 8.0,
              "value": 0.0
            }
          ],
          "translate": [
            {
              "x": 0.0,
              "y": 0.0,
              "curve": [0.04167, 0.0, 0.08333, 0.0, 0.04167, 0.1, 0.08333, 0.2]
            },
            {
              "time": 8.0,
              "x": 0.0,
              "y": 0.0
            }
          ],
          "scale": [
            {
              "x": 1.0,
              "y": 1.0,
              "curve": [0.04167, 1.0, 0.08333, 1.0, 0.04167, 1.001, 0.08333, 1.002]
            },
            {
              "time": 8.0,
              "x": 1.0,
              "y": 1.0
            }
          ]
        }
      }
    }
  }
}
```

### Invariants for T3 (`src/player/`) Alignment
1. `skeleton.spine` is always `"4.2.43"`.
2. `skeleton.images` is `"./images/"`, and every slot attachment image is a 4px-transparent-padded RGBA PNG at `images/<attachment>.png`.
3. `bones` are topologically ordered (parent always precedes child) with `inherit: "normal"` (default).
4. `ik` constraints are sorted by `order` ascending, support 1-bone (`len(bones) == 1`) and 2-bone (`len(bones) == 2`) chains, `mix == 1.0`, no softness/stretch/compress.
5. `slots` are ordered back-to-front (`drawOrder` ascending).
6. Weighted mesh `vertices` use Spine 4.2 packed format: for each vertex `k in 0..N-1`, `[boneCount, boneIndex, localX, localY, weight, ...]` with weights summing to `1.0`.
7. `animations.idle` is a seamless periodic loop (default `8.0s`) where every bone timeline equals setup identity at `t = 0.0` and `t = duration` (`rotate.value == 0`, `translate.x == 0, y == 0`, `scale.x == 1, y == 1`).
8. Cubic Bezier `curve` arrays contain `4 * numFields` floats `[cx1, cy1, cx2, cy2, ...]` in absolute `(time, value)` coordinates.

---

## 2. `parts.json`

```json
{
  "version": "1.0",
  "source": "portrait.png",
  "sourceSHA256": "<64-hex>",
  "canvas": [812, 1446],
  "opaqueThreshold": 250,
  "parts": [
    {
      "name": "back_hair",
      "drawOrder": 0,
      "kind": "soft",
      "parent": "torso",
      "pivot": [300.0, 200.0],
      "bbox": [110, 105, 590, 735],
      "image": "parts/back_hair.png",
      "centerline": [[300.0, 200.0], [630.0, 620.0]],
      "joints": {},
      "notes": ""
    }
  ]
}
```

---

## 3. `rig-build-report.json`

```json
{
  "revision": "portrait-spine-studio-v1",
  "spineVersion": "4.2.43",
  "character": "character_name",
  "spineFile": "character_name.spine.json",
  "preset": "agile_full",
  "gain": 1.0,
  "duration": 8.0,
  "fps": 30,
  "canvas": [812, 1446],
  "origin": [406.0, 1446.0],
  "padPx": 4,
  "partsDir": "/abs/path/to/parts",
  "partsJsonSHA256": "<64-hex>",
  "boneCount": 32,
  "slotCount": 18,
  "ikCount": 4,
  "ik": [],
  "attachments": [
    {"part": "head", "attachment": "region", "recipe": "region", "bone": "head"}
  ],
  "frame0": "setup; all channel offsets zero at 0 and duration",
  "softReferenceRanges": {
    "head": [4.0, 28.0],
    "chest": [3.0, 20.0],
    "hips": [2.0, 22.0],
    "hair_tip": [8.0, 65.0]
  }
}
```

---

## 4. `health-report.json`

```json
{
  "version": "1.0",
  "status": "pass",
  "hardGates": {
    "rest_diff": {
      "passed": true,
      "threshold": {"maxRGBA": 2},
      "measured": {"maxRGBA": 0, "meanAbsDiff": 0.0, "pixelsOver2": 0}
    },
    "setup_identity": {
      "passed": true,
      "threshold": {"frame0VsSetup": 0, "frameLoopEndVsSetup": 0, "setupVsPartsMaxRGBA": 2},
      "measured": {"frame0VsSetup": 0, "frameLoopEndVsSetup": 0, "setupVsPartsMaxRGBA": 0}
    },
    "no_holes": {
      "passed": true,
      "threshold": {"enclosedHolesMax": 0},
      "measured": {"loopEnclosedHolesMax": 0, "stressEnclosedHolesMax": 0, "enclosedHolesMax": 0}
    },
    "anchor_stability": {
      "passed": true,
      "threshold": {"maxDriftPx": 1.5},
      "measured": {"maxDriftPx": 0.01, "perConstraint": {}}
    }
  },
  "softHints": {
    "preset": "agile_full",
    "gain": 1.0,
    "bodyHeightPx": 1200,
    "landmarks": {
      "head": {"ptpPx": 12.4, "refRangePx": [4.0, 28.0], "status": "ok"},
      "chest": {"ptpPx": 8.1, "refRangePx": [3.0, 20.0], "status": "ok"},
      "hips": {"ptpPx": 6.5, "refRangePx": [2.0, 22.0], "status": "ok"},
      "hair_tip": {"ptpPx": 24.0, "refRangePx": [8.0, 65.0], "status": "ok"}
    }
  },
  "stressTest": {
    "gain": 1.6,
    "sampleCount": 8,
    "enclosedHolesMax": 0,
    "perSampleHoles": [0, 0, 0, 0, 0, 0, 0, 0]
  }
}
```

---

## 5. `stage-error.json` (Fail-Stop Artifact)

```json
{
  "status": "failed",
  "failedStage": "inpaint_occluded",
  "errorCode": "MIVO_SERVICE_UNAVAILABLE",
  "message": "Actionable diagnostic message",
  "retryable": true,
  "details": {},
  "preservedIntermediates": {
    "preprocessed": "/abs/path/work/preprocessed.png",
    "matted": "/abs/path/work/matted.png"
  },
  "recoveryActions": ["retry", "discard"]
}
```
