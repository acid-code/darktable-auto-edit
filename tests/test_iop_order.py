"""Second instances need a full iop_order_list (darktable 5.6.2, real sidecar).

A stock darktable sidecar has ``iop_order_version="5"`` and no list. When the
duck plan stacked two colour balance rgb steps, darktable-cli logged "cannot
get iop-order for colorbalancergb instance 1" and "can't find history entry
5 while adding mask" and dropped the background darken with its masks.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from ai_dt.drawn_mask import _write_pipeline_list, splice_shapes
from ai_dt.iop_order import ORDERS
from ai_dt.mapping import exposure_params

FIXTURE = Path(__file__).parent / "fixtures" / "dt562_jpeg.xmp"
PATH = {"type": "path", "points": [[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]], "border": 0.01}


def _list(document: str) -> list[tuple[str, int]]:
    match = re.search(r'darktable:iop_order_list="([^"]*)"', document)
    if match is None:
        return []
    parts = match.group(1).split(",")
    return [(name, int(prio)) for name, prio in zip(parts[0::2], parts[1::2])]


def _cb(document: str, name: str, invert: bool = False) -> str:
    from ai_dt import mask_steps

    step = mask_steps.STEPS["mask_animal_background" if invert else "mask_animal"]
    params, version = mask_steps.module_blob(step, mask_steps.capped_grade(step, "darken" if invert else "lift"))
    return splice_shapes(document, forms=[PATH], operation="colorbalancergb", params=params, modversion=version, name=name, invert=invert)


class IopOrderTests(unittest.TestCase):
    def test_the_tables_match_darktable_5_6(self) -> None:
        self.assertEqual(sorted(ORDERS), [1, 2, 3, 4, 5])
        for version, order in ORDERS.items():
            self.assertEqual(len(order), 93, version)
            self.assertEqual(order[0], "rawprepare")
            self.assertEqual(order[-1], "gamma")
            self.assertEqual(len(set(order)), 93)
            self.assertIn("colorbalancergb", order)

    def test_a_first_instance_writes_no_list(self) -> None:
        base = FIXTURE.read_text(encoding="utf-8")
        self.assertNotIn("iop_order_list", base)
        self.assertIn('darktable:iop_order_version="5"', base)
        once = _cb(base, "Lift the animal")
        self.assertNotIn("iop_order_list", once)

    def test_a_second_instance_writes_the_full_v5_list(self) -> None:
        base = FIXTURE.read_text(encoding="utf-8")
        twice = _cb(_cb(base, "Lift the animal"), "Quiet around the animals", invert=True)
        self.assertIn('darktable:multi_priority="1"', twice)
        self.assertIn('darktable:iop_order_version="5"', twice)
        slots = _list(twice)
        self.assertEqual(len(slots), 94)
        self.assertEqual([name for name, _ in slots if not _], list(ORDERS[5]))
        at = slots.index(("colorbalancergb", 0))
        self.assertEqual(slots[at + 1], ("colorbalancergb", 1))
        self.assertEqual(slots[-1], ("gamma", 0))
        # A third instance goes into the list that is now there.
        thrice = _cb(twice, "Lift the animal")
        slots = _list(thrice)
        self.assertEqual(slots[at + 2], ("colorbalancergb", 2))
        self.assertEqual(len(re.findall("iop_order_list", thrice)), 1)

    def test_earlier_second_instances_are_listed_too(self) -> None:
        base = FIXTURE.read_text(encoding="utf-8")
        row = (
            '<rdf:li darktable:num="9" darktable:operation="exposure" darktable:enabled="1" '
            'darktable:modversion="6" darktable:multi_priority="2"/>'
        )
        document = base.replace("</rdf:Seq>\n   </darktable:history>", row + "</rdf:Seq>\n   </darktable:history>", 1)
        if row not in document:
            document = re.sub(r"(<darktable:history>\s*<rdf:Seq>)", r"\1" + row, base, count=1)
        listed = _list(_write_pipeline_list(document, "colorbalancergb", 1))
        self.assertIn(("exposure", 2), listed)
        self.assertIn(("colorbalancergb", 1), listed)
        self.assertLess(listed.index(("exposure", 0)), listed.index(("exposure", 2)))

    def test_a_sidecar_without_a_version_is_left_alone(self) -> None:
        document = '<x:xmpmeta><rdf:Description darktable:history_end="1"/></x:xmpmeta>'
        self.assertEqual(_write_pipeline_list(document, "exposure", 1), document)
        self.assertTrue(exposure_params(0.1))


if __name__ == "__main__":
    unittest.main()
