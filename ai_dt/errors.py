"""Failures the pipeline can recover from or report clearly."""


class DetectorUnavailable(RuntimeError):
    """A requested detector cannot run in this environment."""


class AdvisorUnavailable(RuntimeError):
    """A requested lighting advisor cannot run in this environment."""


class RawSupportUnavailable(RuntimeError):
    """A camera RAW was requested, but the RAW decoder is not installed."""


class RawReadError(RuntimeError):
    """A camera RAW file could not be decoded."""
