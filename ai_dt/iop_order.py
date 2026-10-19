"""darktable's built-in pipeline orders (``src/common/iop_order.c``, release-5.6.2).

A sidecar names one of these by ``darktable:iop_order_version`` (1 legacy,
2 v3.0, 3 v3.0 JPEG, 4 v5.0, 5 v5.0 JPEG). The tables only hold instance 0
of each module. darktable writes ``darktable:iop_order_list`` as soon as a
module has a second instance; without it, reading the sidecar fails with
"cannot get iop-order for <op> instance 1" and drops that history row and
its masks. Regenerate from darktable's source when it adds modules.
"""

from __future__ import annotations

ORDERS: dict[int, tuple[str, ...]] = {
    # legacy_order
    1: (
        "rawprepare", "invert", "temperature", "rasterfile", "highlights", "cacorrect",
        "hotpixels", "rawdenoise", "demosaic", "mask_manager", "denoiseprofile", "tonemap",
        "exposure", "spots", "retouch", "lens", "cacorrectrgb", "ashift",
        "liquify", "rotatepixels", "scalepixels", "flip", "enlargecanvas", "clipping",
        "toneequal", "crop", "overlay", "graduatednd", "basecurve", "bilateral",
        "profile_gamma", "hazeremoval", "colorin", "channelmixerrgb", "diffuse", "censorize",
        "negadoctor", "blurs", "basicadj", "primaries", "colorreconstruct", "colorchecker",
        "defringe", "equalizer", "vibrance", "colorharmonizer", "colorbalance", "colorequal",
        "colorbalancergb", "colorize", "colortransfer", "colormapping", "bloom", "nlmeans",
        "globaltonemap", "shadhi", "atrous", "bilat", "colorzones", "lowlight",
        "monochrome", "sigmoid", "agx", "filmic", "filmicrgb", "colisa",
        "zonesystem", "tonecurve", "levels", "rgblevels", "rgbcurve", "relight",
        "colorcorrection", "sharpen", "lowpass", "highpass", "grain", "lut3d",
        "colorcontrast", "colorout", "channelmixer", "soften", "vignette", "splittoning",
        "velvia", "clahe", "finalscale", "overexposed", "rawoverexposed", "dither",
        "borders", "watermark", "gamma",
    ),
    # v30_order
    2: (
        "rawprepare", "invert", "temperature", "rasterfile", "highlights", "cacorrect",
        "hotpixels", "rawdenoise", "demosaic", "denoiseprofile", "bilateral", "rotatepixels",
        "scalepixels", "lens", "cacorrectrgb", "hazeremoval", "ashift", "flip",
        "enlargecanvas", "overlay", "clipping", "liquify", "spots", "retouch",
        "exposure", "mask_manager", "tonemap", "toneequal", "crop", "graduatednd",
        "profile_gamma", "equalizer", "colorin", "channelmixerrgb", "diffuse", "censorize",
        "negadoctor", "blurs", "primaries", "nlmeans", "colorchecker", "defringe",
        "atrous", "lowpass", "highpass", "sharpen", "colortransfer", "colormapping",
        "channelmixer", "basicadj", "colorharmonizer", "colorbalance", "colorequal", "colorbalancergb",
        "rgbcurve", "rgblevels", "basecurve", "filmic", "sigmoid", "agx",
        "filmicrgb", "lut3d", "colisa", "tonecurve", "levels", "shadhi",
        "zonesystem", "globaltonemap", "relight", "bilat", "colorcorrection", "colorcontrast",
        "velvia", "vibrance", "colorzones", "bloom", "colorize", "lowlight",
        "monochrome", "grain", "soften", "splittoning", "vignette", "colorreconstruct",
        "colorout", "clahe", "finalscale", "overexposed", "rawoverexposed", "dither",
        "borders", "watermark", "gamma",
    ),
    # v30_jpg_order
    3: (
        "rawprepare", "invert", "temperature", "rasterfile", "highlights", "cacorrect",
        "hotpixels", "rawdenoise", "demosaic", "colorin", "denoiseprofile", "bilateral",
        "rotatepixels", "scalepixels", "lens", "cacorrectrgb", "hazeremoval", "ashift",
        "flip", "enlargecanvas", "overlay", "clipping", "liquify", "spots",
        "retouch", "exposure", "mask_manager", "tonemap", "toneequal", "crop",
        "graduatednd", "profile_gamma", "equalizer", "channelmixerrgb", "diffuse", "censorize",
        "negadoctor", "blurs", "primaries", "nlmeans", "colorchecker", "defringe",
        "atrous", "lowpass", "highpass", "sharpen", "colortransfer", "colormapping",
        "channelmixer", "basicadj", "colorharmonizer", "colorbalance", "colorequal", "colorbalancergb",
        "rgbcurve", "rgblevels", "basecurve", "filmic", "agx", "sigmoid",
        "filmicrgb", "lut3d", "colisa", "tonecurve", "levels", "shadhi",
        "zonesystem", "globaltonemap", "relight", "bilat", "colorcorrection", "colorcontrast",
        "velvia", "vibrance", "colorzones", "bloom", "colorize", "lowlight",
        "monochrome", "grain", "soften", "splittoning", "vignette", "colorreconstruct",
        "colorout", "clahe", "finalscale", "overexposed", "rawoverexposed", "dither",
        "borders", "watermark", "gamma",
    ),
    # v50_order
    4: (
        "rawprepare", "invert", "temperature", "rasterfile", "highlights", "cacorrect",
        "hotpixels", "rawdenoise", "demosaic", "denoiseprofile", "bilateral", "rotatepixels",
        "scalepixels", "lens", "cacorrectrgb", "hazeremoval", "ashift", "flip",
        "enlargecanvas", "overlay", "clipping", "liquify", "spots", "retouch",
        "exposure", "mask_manager", "tonemap", "toneequal", "crop", "graduatednd",
        "profile_gamma", "equalizer", "colorin", "channelmixerrgb", "diffuse", "censorize",
        "negadoctor", "blurs", "primaries", "nlmeans", "colorchecker", "defringe",
        "atrous", "lowpass", "highpass", "sharpen", "colortransfer", "colormapping",
        "channelmixer", "basicadj", "colorharmonizer", "colorbalance", "colorequal", "colorbalancergb",
        "rgbcurve", "rgblevels", "basecurve", "filmic", "sigmoid", "agx",
        "filmicrgb", "lut3d", "colisa", "tonecurve", "levels", "shadhi",
        "zonesystem", "globaltonemap", "relight", "bilat", "colorcorrection", "colorcontrast",
        "velvia", "vibrance", "colorzones", "bloom", "colorize", "lowlight",
        "monochrome", "grain", "soften", "splittoning", "vignette", "colorreconstruct",
        "finalscale", "colorout", "clahe", "overexposed", "rawoverexposed", "dither",
        "borders", "watermark", "gamma",
    ),
    # v50_jpg_order
    5: (
        "rawprepare", "invert", "temperature", "rasterfile", "highlights", "cacorrect",
        "hotpixels", "rawdenoise", "demosaic", "colorin", "denoiseprofile", "bilateral",
        "rotatepixels", "scalepixels", "lens", "cacorrectrgb", "hazeremoval", "ashift",
        "flip", "enlargecanvas", "overlay", "clipping", "liquify", "spots",
        "retouch", "exposure", "mask_manager", "tonemap", "toneequal", "crop",
        "graduatednd", "profile_gamma", "equalizer", "channelmixerrgb", "diffuse", "censorize",
        "negadoctor", "blurs", "primaries", "nlmeans", "colorchecker", "defringe",
        "atrous", "lowpass", "highpass", "sharpen", "colortransfer", "colormapping",
        "channelmixer", "basicadj", "colorharmonizer", "colorbalance", "colorequal", "colorbalancergb",
        "rgbcurve", "rgblevels", "basecurve", "filmic", "sigmoid", "agx",
        "filmicrgb", "lut3d", "colisa", "tonecurve", "levels", "shadhi",
        "zonesystem", "globaltonemap", "relight", "bilat", "colorcorrection", "colorcontrast",
        "velvia", "vibrance", "colorzones", "bloom", "colorize", "lowlight",
        "monochrome", "grain", "soften", "splittoning", "vignette", "colorreconstruct",
        "finalscale", "colorout", "clahe", "overexposed", "rawoverexposed", "dither",
        "borders", "watermark", "gamma",
    ),
}
