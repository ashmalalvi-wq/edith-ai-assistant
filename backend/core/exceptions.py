"""Shared exception types used across E.D.I.T.H's backend."""


class EdithError(Exception):
    """Base class for all application-specific errors."""

    def __init__(self, message: str, *, status_code: int = 500):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class ProviderNotAvailableError(EdithError):
    """Raised when a requested AI provider is disabled or unreachable."""

    def __init__(self, provider: str, detail: str = ""):
        message = f"AI provider '{provider}' is not available." + (f" {detail}" if detail else "")
        super().__init__(message, status_code=503)


class ProviderNotImplementedError(EdithError):
    """Raised when a requested AI provider exists but isn't implemented yet."""

    def __init__(self, provider: str):
        super().__init__(
            f"AI provider '{provider}' is a placeholder and is not implemented yet.",
            status_code=501,
        )


class ModelRequestError(EdithError):
    """Raised when a provider fails while generating a response."""

    def __init__(self, provider: str, detail: str):
        super().__init__(f"Request to provider '{provider}' failed: {detail}", status_code=502)


class NotFoundError(EdithError):
    """Raised when a requested record (project, conversation, memory, task) doesn't exist."""

    def __init__(self, resource: str, resource_id: str):
        super().__init__(f"{resource} '{resource_id}' was not found.", status_code=404)


class VectorStoreError(EdithError):
    """Raised when Qdrant is unreachable or a vector operation fails."""

    def __init__(self, detail: str):
        super().__init__(f"Vector store error: {detail}", status_code=502)


class EmbeddingError(EdithError):
    """Raised when the embedding provider fails to generate a vector."""

    def __init__(self, detail: str):
        super().__init__(f"Embedding generation failed: {detail}", status_code=502)


class ToolNotFoundError(EdithError):
    """Raised when a requested tool name isn't registered."""

    def __init__(self, tool_name: str):
        super().__init__(f"Tool '{tool_name}' is not registered.", status_code=404)


class ToolDisabledError(EdithError):
    """Raised when a registered tool has been disabled."""

    def __init__(self, tool_name: str):
        super().__init__(f"Tool '{tool_name}' is disabled.", status_code=403)


class PermissionDeniedError(EdithError):
    """Raised when a tool's permission policy is 'never_allow'."""

    def __init__(self, tool_name: str):
        super().__init__(f"Tool '{tool_name}' is set to never allow execution.", status_code=403)


class ConfirmationRequiredError(EdithError):
    """Raised when a tool's permission policy is 'ask_every_time' and the caller didn't confirm."""

    def __init__(self, tool_name: str):
        super().__init__(
            f"Tool '{tool_name}' requires confirmation before it can run.", status_code=409
        )


class SandboxViolationError(EdithError):
    """Raised when a filesystem/terminal/VS Code action targets a path outside approved directories."""

    def __init__(self, path: str):
        super().__init__(f"Path '{path}' is outside E.D.I.T.H's approved directories.", status_code=403)


class ToolExecutionError(EdithError):
    """Raised when a tool's own logic fails during execution."""

    def __init__(self, tool_name: str, detail: str):
        super().__init__(f"Tool '{tool_name}' failed: {detail}", status_code=502)


class HostUnavailableError(EdithError):
    """Raised when a host-touching tool is called from a non-host-mode backend (e.g. Dockerized)."""

    def __init__(self, tool_name: str):
        super().__init__(
            f"Tool '{tool_name}' requires host access, but this backend isn't running in host mode "
            "(HOST_MODE=false). Run the backend natively to use this tool.",
            status_code=503,
        )
