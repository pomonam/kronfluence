class FactorsNotFoundError(ValueError):
    """Exception raised when influence factors are not found."""


class TrackedModuleNotFoundError(ValueError):
    """Exception raised when the tracked module is not found."""


class IllegalTaskConfigurationError(ValueError):
    """Exception raised when the provided task is determined to be invalid."""


class UnsupportableModuleError(NotImplementedError):
    """Exception raised when the provided module is not supported."""
