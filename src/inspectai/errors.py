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
