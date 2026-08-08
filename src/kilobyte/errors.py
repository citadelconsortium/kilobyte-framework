class KilobyteError(Exception):
    """Base error suitable for display to the user."""


class ConfigurationError(KilobyteError):
    pass


class ModelUnavailable(KilobyteError):
    pass


class RuntimeUnavailable(KilobyteError):
    pass


class SecurityError(KilobyteError):
    pass


class PermissionDenied(SecurityError):
    pass


class ToolError(KilobyteError):
    pass

