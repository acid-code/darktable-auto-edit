"""Animal outline geometry: the outline must hug the real edge.

darktable feathers a path outward from its corners (``border * min(w, h)``),
so an outline on the segmented edge made the lift spill onto the water and
left an inverted (background) mask a ring short of full strength: a bright
halo. These tests pin the inset, the corner limit, holes in subtractions and
the halo metric of tools/verify_dt.py. No network, no darktable.
"""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ai_dt import drawn_mask, dt_shapes, mask_steps
from ai_dt.segment.animals import ANIMAL_BORDER, Animal, AnimalResult, _holes, animal_forms, inset_mask

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("verify_dt_outline", ROOT / "tools" / "verify_dt.py")
verify_dt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(verify_dt)

W, H = 800, 600


def _disk(size: int, radius: float) -> np.ndarray:
    yy, xx = np.mgrid[:size, :size]
    return np.hypot(xx - size / 2, yy - size / 2) <= radius


def _duck_mask() -> np.ndarray:
    """A body over two legs with water between them, full frame."""

    image = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(image)
    draw.ellipse((330, 120, 470, 430), fill=255)
    draw.rectangle((355, 400, 367, 540), fill=255)  # left leg, 13 px
    draw.rectangle((433, 400, 445, 540), fill=255)  # right leg
    draw.rectangle((340, 530, 460, 548), fill=255)  # feet on the ledge
    return np.asarray(image) > 127


def _result(mask: np.ndarray) -> AnimalResult:
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = xs.min() - 8, xs.max() + 9, ys.min() - 8, ys.max() + 9
    crop = (x0 / W, y0 / H, (x1 - x0) / W, (y1 - y0) / H)
    animal = Animal("duck", 0.95, crop, mask.astype(np.float32), "test", crop, mask[y0:y1, x0:x1].astype(np.float32))
    return AnimalResult(animals=[animal])


def _fill(points: list, width: int = W, height: int = H) -> np.ndarray:
    image = Image.new("L", (width, height), 0)
    ImageDraw.Draw(image).polygon([(x * width, y * height) for x, y in points], fill=255)
    return np.asarray(image) > 127


class InsetTests(unittest.TestCase):
    def test_a_disk_shrinks_by_about_the_radius(self) -> None:
        shrunk = inset_mask(_disk(200, 60), 8)
        radius = np.sqrt(shrunk.sum() / np.pi)
        self.assertAlmostEqual(radius, 52, delta=2.0)

    def test_a_thin_leg_is_kept_whole(self) -> None:
        mask = _disk(200, 50)
        mask[100:195, 97:103] = True  # 6 px leg, thinner than twice the inset
        out = inset_mask(mask, 8)
        self.assertTrue(out[160:190, 97:103].all())

    def test_a_hole_is_not_grown(self) -> None:
        mask = _disk(200, 60)
        hole = _disk(200, 20)
        mask &= ~hole
        out = inset_mask(mask, 8)
        self.assertTrue(np.array_equal(_holes(out), hole))
        self.assertAlmostEqual(np.sqrt((out | hole).sum() / np.pi), 52, delta=2.0)

    def test_no_radius_changes_nothing(self) -> None:
        mask = _disk(64, 20)
        self.assertTrue(np.array_equal(inset_mask(mask, 0.2), mask))


class OutlineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mask = _duck_mask()
        self.forms = animal_forms(_result(self.mask))

    def test_corners_stay_under_the_writer_limit(self) -> None:
        for form in self.forms:
            self.assertLessEqual(len(form["points"]), dt_shapes.MAX_PATH_POINTS)
        many = animal_forms(_result(self.mask), max_points=200)
        self.assertTrue(all(len(form["points"]) <= dt_shapes.MAX_PATH_POINTS for form in many))

    def test_the_outline_sits_inside_the_edge_by_the_feather(self) -> None:
        outline = _fill(self.forms[0]["points"])
        for hole in self.forms[1:]:
            outline &= ~_fill(hole["points"])
        body = self.mask.copy()
        body[395:] = False  # the legs keep their own outline; check the body
        outline_body = outline.copy()
        outline_body[395:] = False
        self.assertLess((outline & ~self.mask).sum() / outline.sum(), 0.01)
        # Feather is border * min(W, H) = 3 px here; the body ring between
        # outline and edge is about that wide.
        ring = (body & ~outline_body).sum()
        perimeter = np.pi * (70 + 155) * 1.1
        self.assertAlmostEqual(ring / perimeter, ANIMAL_BORDER * min(W, H), delta=1.5)

    def test_legs_and_feet_are_in_the_outline(self) -> None:
        outline = _fill(self.forms[0]["points"])
        for x in (361, 439):
            self.assertGreater(outline[440:520, x].mean(), 0.9)
        self.assertGreater(outline[537:542, 350:450].mean(), 0.8)

    def test_the_water_between_the_legs_is_a_hole(self) -> None:
        holes = [form for form in self.forms if form["mode"] == "difference"]
        self.assertEqual(len(holes), 1)
        inside = _fill(holes[0]["points"])
        self.assertTrue(inside[470, 400])
        # The hole is traced on the real edge, not grown into the legs.
        self.assertLess((inside & self.mask).sum() / inside.sum(), 0.05)
        water = _holes(self.mask)
        self.assertGreater((inside & water).sum() / water.sum(), 0.9)


class SubtractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.animal = animal_forms(_result(_duck_mask()))

    def test_a_gradient_minus_the_animal_keeps_the_water_between_its_legs(self) -> None:
        base = mask_steps.bottom_gradient_forms(extent=0.4)
        forms = mask_steps._subtract(base, self.animal)
        self.assertEqual([form["mode"] for form in forms][-2:], ["union", "intersect"])
        gradient = dt_shapes.render_forms(base, W, H)
        cut = dt_shapes.render_forms(forms, W, H)
        self.assertAlmostEqual(float(cut[470, 400]), float(gradient[470, 400]), delta=0.02)
        self.assertLess(float(cut[300, 400]), 0.02)  # the body is cut
        self.assertAlmostEqual(float(cut[580, 100]), float(gradient[580, 100]), delta=0.01)

    def test_several_base_forms_add_the_hole_back_at_their_strength(self) -> None:
        forms = mask_steps.edge_burn_forms(self.animal, extent=0.3)
        added = [form for form in forms if form.get("mode") == "union" and form.get("type") == "path"]
        self.assertEqual(len(added), 1)
        self.assertGreater(added[0]["opacity"], 0.0)
        base = dt_shapes.render_forms([form for form in forms if form.get("type") == "gradient"], W, H)
        cut = dt_shapes.render_forms(forms, W, H)
        self.assertAlmostEqual(float(cut[470, 400]), float(base[470, 400]), delta=0.15)

    def test_no_holes_no_extra_members(self) -> None:
        outer = [form for form in self.animal if form["mode"] != "difference"]
        forms = mask_steps._subtract(mask_steps.bottom_gradient_forms(), outer)
        self.assertEqual([form["mode"] for form in forms], ["union", "difference"])


class WriterTests(unittest.TestCase):
    def test_a_long_path_is_thinned_evenly_not_cut(self) -> None:
        angles = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        points = [[0.5 + 0.3 * np.cos(a), 0.5 + 0.3 * np.sin(a)] for a in angles]
        _kind, blob, count = drawn_mask._form_record({"type": "path", "points": points, "border": 0.005})
        self.assertEqual(count, dt_shapes.MAX_PATH_POINTS)
        corners = dt_shapes.decode_path(blob)
        xs = [p.x for p in corners]
        ys = [p.y for p in corners]
        # A cut list would close with a chord and lose the last third.
        self.assertLess(min(xs), 0.21)
        self.assertGreater(max(xs), 0.79)
        self.assertLess(min(ys), 0.21)


    def test_traced_outlines_keep_sharp_handles_at_sharp_turns(self) -> None:
        square = [(0.2, 0.2), (0.5, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]
        plain = dt_shapes.smooth_path(square)
        traced = dt_shapes.smooth_path(square, sharp_turns=True)
        self.assertNotEqual(plain[3].ctrl1, (0.8, 0.8))  # default unchanged
        self.assertEqual(traced[3].ctrl1, (0.8, 0.8))
        self.assertEqual(traced[3].ctrl2, (0.8, 0.8))
        self.assertNotEqual(traced[1].ctrl1, (0.5, 0.2))  # a straight run stays smooth
        self.assertTrue(all(form.get("sharp_turns") for form in animal_forms(_result(_duck_mask()))))


class HaloMetricTests(unittest.TestCase):
    def _pngs(self, ring_change: float, far_change: float, inside_change: float = 0.0):
        tmp = Path(tempfile.mkdtemp())
        outline = _disk(200, 50)
        dist = np.zeros((200, 200))
        yy, xx = np.mgrid[:200, :200]
        dist = np.hypot(xx - 100, yy - 100) - 50
        base = np.full((200, 200), 120.0)
        test = base + np.where(dist <= 0, inside_change, np.where(dist <= 10, ring_change, far_change))
        Image.fromarray(base.astype(np.uint8)).save(tmp / "base.png")
        Image.fromarray(np.clip(test, 0, 255).astype(np.uint8)).save(tmp / "test.png")
        return verify_dt.halo_metric(tmp / "base.png", tmp / "test.png", outline)

    def test_an_even_darken_is_no_halo(self) -> None:
        metric = self._pngs(-16, -16)
        self.assertAlmostEqual(metric["ring"], -16 / 120 * 100, delta=0.5)  # percent of the base
        self.assertEqual(verify_dt.halo_failure("match", metric), "")

    def test_a_ring_short_of_the_darken_is_a_halo(self) -> None:
        # The old duck outline: ring -12.9 against -16.1 around it (luma).
        metric = self._pngs(-12.9, -16.1)
        self.assertIn("ring", verify_dt.halo_failure("match", metric))

    def test_a_lift_on_the_ring_is_a_spill(self) -> None:
        self.assertIn("spills", verify_dt.halo_failure("spill", self._pngs(4.8, 0, 10)))
        self.assertEqual(verify_dt.halo_failure("spill", self._pngs(0.1, 0, 10)), "")


if __name__ == "__main__":
    unittest.main()
