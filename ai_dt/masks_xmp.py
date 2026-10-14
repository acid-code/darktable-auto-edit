"""Spike: can an XMP sidecar carry the ellipses already in ``local_edits``?

Darktable's ``image:apply_sidecar()`` replaces the whole history stack.
A ``.dtstyle`` file still cannot store the mask table those ellipses would
need. Writing the ellipse into a sidecar and then applying that sidecar
would drop the photographer's modules. Until there is a way to attach a
mask without replacing history, local stages stay in ``edited.tif``.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET


MASKS_REASON = (
    "A sidecar would replace the whole history, and a style file cannot "
    "store the mask table. Local stages stay in edited.tif."
)


def masks_can_attach() -> bool:
    """Whether a drawn mask can be added without replacing history."""

    return False


def ellipse_sidecar(local_edits: list[dict]) -> str:
    """An XMP note of the ellipses. It is not applied to the image."""

    root = ET.Element("xmpmeta")
    note = ET.SubElement(root, "ai_editor_masks")
    ET.SubElement(note, "attached").text = "0"
    ET.SubElement(note, "reason").text = MASKS_REASON
    edits = ET.SubElement(note, "ellipses")
    for edit in local_edits:
        if not isinstance(edit, dict):
            continue
        ellipse = edit.get("ellipse")
        if not isinstance(ellipse, dict):
            continue
        node = ET.SubElement(edits, "ellipse")
        node.set("target", str(edit.get("target") or ""))
        for key in ("cx", "cy", "rx", "ry", "feather"):
            if key in ellipse:
                node.set(key, str(ellipse[key]))
    return ET.tostring(root, encoding="unicode")


def sidecar_attaches_mask(document: str) -> bool:
    """True only when the document both describes an ellipse and says it attached."""

    try:
        root = ET.fromstring(document)
    except ET.ParseError:
        return False
    note = root.find("ai_editor_masks")
    if note is None:
        return False
    attached = note.findtext("attached") or "0"
    ellipses = note.find("ellipses")
    has_ellipse = ellipses is not None and any(True for _ in ellipses)
    return attached == "1" and has_ellipse
