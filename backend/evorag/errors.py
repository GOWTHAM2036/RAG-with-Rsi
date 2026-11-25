class EvoRagError(Exception):
    """Base for application-level failures."""


class ConfigurationError(EvoRagError):
    """The supplied configuration could not be read or validated."""


class ResourceUnavailable(EvoRagError):
    """A job could not be admitted within the configured resource budget."""


class IngestionError(EvoRagError):
    """A public error code. Never include document content or a local path."""

    def __init__(self, message: str, status: int = 422, *, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code or message
        self.status = status
        self.job_id: str | None = None
