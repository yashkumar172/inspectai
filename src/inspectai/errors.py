"""Exception types shared across the package."""


class InspectAIError(Exception):
    """Base class for all errors raised deliberately by this package."""


class ConfigError(InspectAIError):
    """A setting or environment variable is missing or invalid."""


class DatasetError(InspectAIError):
    """The dataset or its split file is missing, malformed, or inconsistent."""


class SplitError(InspectAIError):
    """A data split is invalid, for example because its parts overlap."""


class ImageError(InspectAIError):
    """An image or mask cannot be read, or fails a validity check."""


class ModelError(InspectAIError):
    """A model is used before fitting, or is fitted or called with invalid inputs."""


class EvaluationError(InspectAIError):
    """Metrics, thresholds, or reports cannot be computed from the given inputs."""


class ArtifactError(InspectAIError):
    """A model artifact is missing, corrupted, stale, or does not match its manifest."""


class WeightsNotFoundError(InspectAIError):
    """Pretrained weights are not in the local cache and may not be downloaded implicitly."""


class UploadError(InspectAIError):
    """An uploaded image is rejected; the message is safe to show to the user."""


class ModelUnavailableError(InspectAIError):
    """No fitted model exists for the requested model and category."""
