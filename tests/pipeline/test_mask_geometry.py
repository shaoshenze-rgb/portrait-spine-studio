"""ISSUE-01: derive rigid pivots and soft centerlines from exclusive masks."""
from __future__ import annotations

import copy
import math
import time
import unittest

import numpy as np

from src.pipeline.mask_geometry import (
    _largest_component,
    derive_contract_geometry,
    derive_rigid_pivot,
    derive_soft_centerline,
)


def _in_mask(mask: np.ndarray, x: float, y: float) -> bool:
    h, w = mask.shape
    xi = int(round(x))
    yi = int(round(y))
    return 0 <= yi < h and 0 <= xi < w and bool(mask[yi, xi])


def _paint_disks(shape, centers, radius) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    r2 = float(radius) ** 2
    yy, xx = np.ogrid[:h, :w]
    for x, y in centers:
        mask |= (xx - x) ** 2 + (yy - y) ** 2 <= r2
    return mask


def _polyline_centers(pts, n_per_seg=20):
    out = []
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        for t in np.linspace(0.0, 1.0, n_per_seg, endpoint=False):
            out.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
    out.append(tuple(pts[-1]))
    return out


def _seg_min_dist_to_mask(mask: np.ndarray, x0, y0, x1, y1, samples=8) -> float:
    ys, xs = np.nonzero(mask)
    coords = np.column_stack((xs.astype(np.float64), ys.astype(np.float64)))
    worst = 0.0
    for t in np.linspace(0.0, 1.0, samples):
        x = x0 + t * (x1 - x0)
        y = y0 + t * (y1 - y0)
        d = np.sqrt((coords[:, 0] - x) ** 2 + (coords[:, 1] - y) ** 2).min()
        worst = max(worst, float(d))
    return worst



def _chords_empty_pixels(mask, pts):
    empty = 0
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        dist = math.hypot(x1 - x0, y1 - y0)
        n = max(1, int(math.ceil(dist)))
        for i in range(n + 1):
            t = i / float(n)
            x = x0 + t * (x1 - x0)
            y = y0 + t * (y1 - y0)
            if not _in_mask(mask, x, y):
                empty += 1
    return empty


class MaskGeometryTests(unittest.TestCase):
    def assert_points_in_mask(self, mask, pts, msg=""):
        self.assertGreaterEqual(len(pts), 3)
        for i, (x, y) in enumerate(pts):
            self.assertTrue(
                _in_mask(mask, x, y),
                "%s point[%d]=(%.3f, %.3f) not in mask" % (msg, i, x, y),
            )

    def test_straight_strip_root_to_tip_equidistant(self):
        h, w = 200, 80
        mask = np.zeros((h, w), dtype=bool)
        mask[20:180, 36:44] = True
        parent = np.zeros((h, w), dtype=bool)
        parent[8:22, 28:52] = True
        frozen = mask.copy()
        pts = derive_soft_centerline(mask, parent, num_points=5)
        np.testing.assert_array_equal(mask, frozen)
        self.assertEqual(len(pts), 5)
        self.assert_points_in_mask(mask, pts, "straight")
        ys = [p[1] for p in pts]
        xs = [p[0] for p in pts]
        self.assertLess(ys[0], ys[-1])
        self.assertLess(ys[0], 28.0)
        self.assertGreater(ys[-1], 168.0)
        for x in xs:
            self.assertGreaterEqual(x, 35.0)
            self.assertLessEqual(x, 44.0)
        segs = [math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]) for i in range(4)]
        avg = sum(segs) / 4.0
        self.assertGreater(avg, 25.0)
        for s in segs:
            self.assertLess(abs(s - avg), 12.0)

    def test_c_curve_geodesic_does_not_shortcut(self):
        h, w = 160, 160
        yy, xx = np.ogrid[:h, :w]
        cx, cy = 80.0, 80.0
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        ang = np.arctan2(yy - cy, xx - cx)
        mask = (dist >= 28.0) & (dist <= 40.0) & (np.abs(ang) >= 0.72)
        parent = np.zeros((h, w), dtype=bool)
        parent[36:50, 68:92] = True
        hole = dist < 24.0
        pts = derive_soft_centerline(mask, parent, num_points=6)
        self.assert_points_in_mask(mask, pts, "c-curve")
        for x, y in pts:
            self.assertFalse(bool(hole[int(round(y)), int(round(x))]))
        worst = 0.0
        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
            worst = max(worst, _seg_min_dist_to_mask(mask, x0, y0, x1, y1, samples=10))
        self.assertLess(worst, 6.0, "C-curve polyline shortcuts through empty space (%.2f px)" % worst)
        chord = math.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1])
        arc = sum(
            math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]) for i in range(len(pts) - 1)
        )
        self.assertGreater(arc, 1.35 * chord)

    def test_s_curve_no_fork_no_foldback(self):
        h, w = 190, 130
        t = np.linspace(0.0, 1.0, 280)
        xs = 64.0 + 24.0 * np.sin(2.0 * math.pi * t)
        ys = 18.0 + 150.0 * t
        mask = _paint_disks((h, w), list(zip(xs, ys)), 5.5)
        parent = np.zeros((h, w), dtype=bool)
        parent[10:22, 52:76] = True
        pts = derive_soft_centerline(mask, parent, num_points=7)
        self.assert_points_in_mask(mask, pts, "s-curve")
        y_vals = [p[1] for p in pts]
        self.assertLess(y_vals[0], 30.0)
        self.assertGreater(y_vals[-1], 150.0)
        # y is overall increasing: no fold-back along the hanging S.
        for i in range(len(y_vals) - 1):
            self.assertGreaterEqual(y_vals[i + 1] + 4.0, y_vals[i])
        rounded = [(int(round(p[0])), int(round(p[1]))) for p in pts]
        self.assertEqual(len(rounded), len(set(rounded)))
        # x should oscillate (S), not stay on a straight vertical.
        self.assertGreater(max(p[0] for p in pts) - min(p[0] for p in pts), 18.0)

    def test_jagged_hem_not_pulled_by_spikes(self):
        h, w = 140, 90
        smooth = np.zeros((h, w), dtype=bool)
        smooth[24:100, 32:58] = True
        jagged = smooth.copy()
        for x in range(32, 58, 2):
            jagged[100:102, x] = True
            if x % 4 == 0:
                jagged[22:24, x] = False  # 1px nicks at the top edge, not a new component
        jagged[20:24, 40:50] = True  # keep a clean root pad
        parent = np.zeros((h, w), dtype=bool)
        parent[12:25, 36:54] = True
        pts_s = derive_soft_centerline(smooth, parent, num_points=5)
        pts_j = derive_soft_centerline(jagged, parent, num_points=5)
        self.assert_points_in_mask(jagged, pts_j, "jagged")
        self.assert_points_in_mask(smooth, pts_s, "smooth")
        # 1-2px teeth may own the geodesic tip corner; the chain itself must stay centered.
        for a, b in zip(pts_s[:-1], pts_j[:-1]):
            self.assertLess(math.hypot(a[0] - b[0], a[1] - b[1]), 5.0)
        self.assertLess(pts_j[0][1], pts_j[-1][1])
        self.assertGreater(pts_j[-1][1], 96.0)
        xs_mid = [p[0] for p in pts_j[1:-1]]
        for x in xs_mid:
            self.assertTrue(34.0 <= x <= 56.0)

    def test_wide_short_cape_short_path_fallback(self):
        h, w = 70, 90
        mask = np.zeros((h, w), dtype=bool)
        mask[24:34, 28:58] = True  # 30x10 blob, geodesic to a corner ~18px < 24
        parent = np.zeros((h, w), dtype=bool)
        parent[18:26, 38:48] = True
        pts = derive_soft_centerline(mask, parent, num_points=5)
        self.assertEqual(len(pts), 5)
        self.assert_points_in_mask(mask, pts, "cape")
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        self.assertGreater(max(xs) - min(xs), max(ys) - min(ys) - 1.0)
        for y in ys:
            self.assertTrue(23.0 <= y <= 34.0)

    def test_isolated_child_pivot_and_soft_root_stay_inside(self):
        h, w = 160, 160
        parent = np.zeros((h, w), dtype=bool)
        parent[16:48, 16:48] = True
        child = np.zeros((h, w), dtype=bool)
        child[104:144, 112:132] = True
        pivot = derive_rigid_pivot(child, parent, parent_pivot=(32.0, 32.0))
        self.assertTrue(_in_mask(child, pivot[0], pivot[1]))
        self.assertLess(abs(pivot[0] - 112.0), 2.0)
        self.assertLess(abs(pivot[1] - 104.0), 2.0)
        pts = derive_soft_centerline(child, parent, parent_pivot=(32.0, 32.0), num_points=4)
        self.assert_points_in_mask(child, pts, "isolated-soft")
        self.assertLess(math.hypot(pts[0][0] - 112.0, pts[0][1] - 104.0), 4.0)
        # contact pivot: overlapping band
        arm = np.zeros((h, w), dtype=bool)
        arm[40:70, 44:70] = True  # touches parent at ~y=40-48, x=44-48
        joint = derive_rigid_pivot(arm, parent)
        self.assertTrue(_in_mask(arm, joint[0], joint[1]))
        self.assertTrue(40.0 <= joint[1] <= 52.0)
        self.assertTrue(44.0 <= joint[0] <= 52.0)

    def test_derive_contract_geometry_normalize_manual_and_topo(self):
        h, w = 80, 100
        torso = np.zeros((h, w), dtype=bool)
        torso[18:58, 44:78] = True
        cape = np.zeros((h, w), dtype=bool)
        cape[22:70, 12:30] = True
        modules = [
            {
                "id": "cape",
                "parent": "torso",
                "motion_type": "soft",
                "pivot": [0.11, 0.12],
                "centerline": [[0.12, 0.20], [0.13, 0.40], [0.14, 0.60], [0.15, 0.75], [0.16, 0.88]],
                "pieces": [{"id": "cape_p0"}],
            },
            {
                "id": "torso",
                "parent": None,
                "motion_type": "rigid",
                "pivot": [0.91, 0.92],
                "manual_pivot": True,
                "pieces": [{"id": "torso_p0"}],
            },
        ]
        original = copy.deepcopy(modules)
        out = derive_contract_geometry((w, h), modules, {"cape_p0": cape, "torso_p0": torso})
        self.assertEqual(original, modules)
        self.assertIsNot(out[0], modules[0])
        by_id = {m["id"]: m for m in out}
        self.assertEqual(by_id["torso"]["pivot"], [0.91, 0.92])
        cp = by_id["cape"]["pivot"]
        cl = by_id["cape"]["centerline"]
        self.assertEqual(len(cl), 5)
        px, py = cp[0] * w, cp[1] * h
        self.assertTrue(_in_mask(cape, px, py))
        for p in cl:
            self.assertTrue(0.0 <= p[0] <= 1.0 and 0.0 <= p[1] <= 1.0)
            self.assertTrue(_in_mask(cape, p[0] * w, p[1] * h))
        # preserve_manual=False overwrites even manual_pivot
        out2 = derive_contract_geometry(
            (w, h), modules, {"cape_p0": cape, "torso_p0": torso}, preserve_manual=False
        )
        tpx, tpy = out2[1]["pivot"][0] * w, out2[1]["pivot"][1] * h
        self.assertTrue(_in_mask(torso, tpx, tpy))
        self.assertNotEqual(out2[1]["pivot"], [0.91, 0.92])
        # geometry_override keeps centerline
        manual_cl = [[0.2, 0.3], [0.2, 0.4], [0.2, 0.5]]
        mods3 = copy.deepcopy(modules)
        mods3[0]["manual_centerline"] = True
        mods3[0]["centerline"] = copy.deepcopy(manual_cl)
        out3 = derive_contract_geometry((w, h), mods3, {"cape_p0": cape, "torso_p0": torso})
        self.assertEqual(out3[0]["centerline"], manual_cl)

    def test_large_2400_soft_module_perf_and_accuracy(self):
        h = w = 2400
        mask = np.zeros((h, w), dtype=bool)
        mask[280:2080, 1186:1214] = True
        parent = np.zeros((h, w), dtype=bool)
        parent[250:290, 1160:1240] = True
        derive_soft_centerline(mask[:400, 1100:1300], num_points=5)  # warmup numpy kernels
        t0 = time.perf_counter()
        pts = derive_soft_centerline(mask, parent, num_points=5)
        elapsed = time.perf_counter() - t0
        self.assert_points_in_mask(mask, pts, "large-2400")
        self.assertLess(pts[0][1], 320.0)
        self.assertGreater(pts[-1][1], 2000.0)
        self.assertLess(elapsed, 0.12, "2400 soft module took %.1f ms" % (elapsed * 1000.0))
        modules = [
            {
                "id": "cape",
                "parent": None,
                "motion_type": "soft",
                "pieces": [{"id": "cape_p0"}],
            }
        ]
        out = derive_contract_geometry((w, h), modules, {"cape_p0": mask})
        for p in out[0]["centerline"]:
            self.assertTrue(_in_mask(mask, p[0] * w, p[1] * h))

    def test_block_maxpool_keeps_1px_tail_1200(self):
        mask = np.zeros((1200, 1200), dtype=bool)
        mask[100:400, 500:800] = True
        mask[399:1100, 799] = True  # real 8-connected seam onto the main block
        parent = np.zeros((1200, 1200), dtype=bool)
        parent[80:110, 520:780] = True
        pts = derive_soft_centerline(mask, parent, num_points=5)
        self.assert_points_in_mask(mask, pts, "1px-tail-1200")
        self.assertGreaterEqual(pts[-1][1], 1080.0)
        self.assertGreater(pts[-1][1], pts[0][1])

    def test_block_maxpool_keeps_thin_ribbon_2400(self):
        mask = np.zeros((2400, 2400), dtype=bool)
        mask[200:800, 1000:1600] = True
        mask[799:2200, 1598] = True
        mask[799:2200, 1599] = True  # 2px tail, attached at x=1599
        parent = np.zeros((2400, 2400), dtype=bool)
        parent[170:210, 1100:1500] = True
        pts = derive_soft_centerline(mask, parent, num_points=5)
        self.assert_points_in_mask(mask, pts, "thin-ribbon-2400")
        self.assertGreaterEqual(pts[-1][1], 1080.0)
        self.assertGreater(pts[-1][1], 2000.0)

    def test_4px_gap_does_not_join_separated_blocks(self):
        mask = np.zeros((1200, 1200), dtype=bool)
        mask[100:400, 500:800] = True
        mask[100:400, 804:820] = True
        mask[404:1100, 803:810] = True
        main = _largest_component(mask)
        self.assertEqual(int(main.sum()), 90000)
        np.testing.assert_array_equal(main[100:400, 500:800], np.ones((300, 300), dtype=bool))
        self.assertFalse(np.any(main[100:400, 804:820]))
        self.assertFalse(np.any(main[404:1100, 803:810]))
        parent = np.zeros((1200, 1200), dtype=bool)
        parent[80:110, 520:780] = True
        pts = derive_soft_centerline(mask, parent, num_points=5)
        self.assert_points_in_mask(mask, pts, "4px-gap")
        for x, y in pts:
            self.assertTrue(500.0 <= x < 800.0, "soft chain crossed 4px gap, x=%.1f" % x)
            self.assertTrue(100.0 <= y < 400.0, "soft chain crossed 4px gap, y=%.1f" % y)

    def test_hairpin_slit_does_not_shortcut(self):
        mask = np.zeros((1200, 1200), dtype=bool)
        mask[100:150, 100:900] = True
        mask[100:206, 860:900] = True
        mask[156:206, 100:900] = True
        parent = np.zeros((1200, 1200), dtype=bool)
        parent[90:120, 90:140] = True
        pts = derive_soft_centerline(mask, parent, num_points=5)
        self.assertEqual(len(pts), 5)
        self.assert_points_in_mask(mask, pts, "hairpin")
        self.assertLess(pts[0][0], 200.0)
        self.assertLess(pts[0][1], 155.0)
        self.assertGreater(max(p[0] for p in pts[1:-1]), 800.0)
        self.assertEqual(_chords_empty_pixels(mask, pts), 0)

    def test_vertical_u_chords_stay_in_mask(self):
        mask = np.zeros((820, 820), dtype=bool)
        mask[100:700, 300:308] = True
        mask[100:700, 312:320] = True
        mask[692:700, 300:320] = True
        parent = np.zeros((820, 820), dtype=bool)
        parent[88:112, 295:313] = True
        pts = derive_soft_centerline(mask, parent, num_points=11)
        self.assertEqual(len(pts), 11)
        self.assert_points_in_mask(mask, pts, "vertical-u")
        self.assertEqual(_chords_empty_pixels(mask, pts), 0)
        self.assertGreater(max(p[1] for p in pts), 680.0)

    def _toy_contract_args(self):
        h, w = 40, 40
        body = np.zeros((h, w), dtype=bool)
        body[8:32, 8:32] = True
        cape = np.zeros((h, w), dtype=bool)
        cape[10:34, 2:12] = True
        modules = [
            {
                "id": "cape",
                "parent": "body",
                "motion_type": "soft",
                "pieces": [{"id": "cape_p0"}],
            },
            {
                "id": "body",
                "parent": None,
                "motion_type": "rigid",
                "pieces": [{"id": "body_p0"}],
            },
        ]
        masks = {"cape_p0": cape, "body_p0": body}
        return (w, h), modules, masks

    def test_contract_rejects_missing_parent(self):
        size, modules, masks = self._toy_contract_args()
        modules[0]["parent"] = "ghost"
        with self.assertRaises(ValueError):
            derive_contract_geometry(size, modules, masks)

    def test_contract_rejects_cycle_and_self_loop(self):
        size, modules, masks = self._toy_contract_args()
        modules[0]["parent"] = "body"
        modules[1]["parent"] = "cape"
        with self.assertRaises(ValueError):
            derive_contract_geometry(size, modules, masks)
        size, modules, masks = self._toy_contract_args()
        modules[1]["parent"] = "body"
        with self.assertRaises(ValueError):
            derive_contract_geometry(size, modules, masks)

    def test_contract_rejects_duplicate_module_id(self):
        size, modules, masks = self._toy_contract_args()
        modules[0]["id"] = "body"
        with self.assertRaises(ValueError):
            derive_contract_geometry(size, modules, masks)

    def test_contract_rejects_missing_or_empty_piece_mask(self):
        size, modules, masks = self._toy_contract_args()
        with self.assertRaises(ValueError):
            derive_contract_geometry(size, modules, {"body_p0": masks["body_p0"]})
        size, modules, masks = self._toy_contract_args()
        modules[0]["pieces"] = []
        with self.assertRaises(ValueError):
            derive_contract_geometry(size, modules, masks)
        size, modules, masks = self._toy_contract_args()
        masks["cape_p0"] = np.zeros_like(masks["cape_p0"])
        with self.assertRaises(ValueError):
            derive_contract_geometry(size, modules, masks)

    def test_auto_soft_root_is_consistent_without_moving_manual_pivot(self):
        parent = np.zeros((80, 80), bool); parent[10:20, 10:30] = True
        child = np.zeros_like(parent); child[30:60, 30:50] = True; child[20:22, 20:22] = True
        modules = [
            {'id': 'body', 'motion_type': 'rigid', 'parent': None, 'pieces': [{'id': 'body_p0'}]},
            {'id': 'cloth', 'motion_type': 'soft', 'parent': 'body', 'pieces': [{'id': 'cloth_p0'}]},
        ]
        masks = {'body_p0': parent, 'cloth_p0': child}
        auto = derive_contract_geometry([80, 80], modules, masks, preserve_manual=False)[1]
        self.assertEqual(auto['pivot'], auto['centerline'][0])
        x, y = auto['pivot']; self.assertTrue(child[round(y*80), round(x*80)])
        modules[1].update(pivot=[.25, .25], manual_pivot=True)
        manual = derive_contract_geometry([80, 80], modules, masks, preserve_manual=True)[1]
        self.assertEqual(manual['pivot'], [.25, .25])
        self.assertFalse(np.array_equal(auto['pivot'], manual['pivot']))
        np.testing.assert_array_equal(masks['cloth_p0'], child)

    def test_empty_mask_and_num_points_raise(self):
        empty = np.zeros((12, 12), dtype=bool)
        filled = np.zeros((12, 12), dtype=bool)
        filled[2:10, 3:9] = True
        with self.assertRaises(ValueError):
            derive_rigid_pivot(empty)
        with self.assertRaises(ValueError):
            derive_soft_centerline(empty)
        with self.assertRaises(ValueError):
            derive_soft_centerline(filled, num_points=2)
        with self.assertRaises(ValueError):
            derive_soft_centerline(filled, num_points=0)


if __name__ == "__main__":
    unittest.main()
