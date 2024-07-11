class FactorsNotFoundError(ValueError):
    """Exception raised when influence factors are not found."""


class TrackedModuleNotFoundError(ValueError):
    """Exception raised when a tracked module is not found in the model."""


class IllegalTaskConfigurationError(ValueError):
    """Exception raised when the provided task configuration is determined to be invalid."""


class UnsupportableModuleError(NotImplementedError):
    """Exception raised when the provided module is not supported by the current implementation."""
