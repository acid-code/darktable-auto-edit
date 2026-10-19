---
name: darktable-vignette
description: >-
  Size a Darktable 5.4.1 vignette so the clear hole covers a subject instead of
  flattening into a line. Use when placing a vignette, changing scale or the
  width/height ratio, or when a vignette darkens the whole frame.
---

# Darktable vignette

Vignette version 4 is 44 bytes: `scale`, `falloff_scale`, `brightness`, `saturation`, `center_x`, `center_y`, autoratio int, `whratio`, `shape`, dithering uint, unbound int. Centers are -1..1, with 0 at the middle.

Darktable treats `scale` as the clear zone, in percent of the frame. 100 covers the whole width and height before the ratio shrinks one axis.

`whratio` only shrinks:

- 0 to 1 multiplies the width
- 1 to 2 multiplies the height by `2 - whratio`

A value near 2 is a horizontal line. The darkening then covers the subject. A nearly square box must stay near 1. Do not map "slightly wider" to `1 + 1/aspect`.

For a subject box in normalized `(x, y, w, h)`:

- Expand `w` and `h` a little around the same center so the falloff starts outside the body.
- `scale = 100 * max(w, h)`, clamped to about 18..150. Do not reuse the automatic animal cap of 24. That cap is smaller than two statues, so the hole misses them.
- If `w >= h`, `whratio = 2 - h/w`, and keep it at or below 1.35.
- If `h > w`, `whratio = w/h`, and keep it at or above 0.65.
- `falloff_scale` around 28. `brightness` around -0.32. Negative brightness darkens only outside the hole. It does not lift the inside.

The accept step for a body is this vignette, named `Vignette the subject`. Without a body box the subject step stays an exposure. A sentence with no hue word that asks to lift again still expects another exposure, not a second vignette.
