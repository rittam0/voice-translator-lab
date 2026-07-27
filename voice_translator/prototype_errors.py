from __future__ import annotations


class PrototypeError(RuntimeError):
    """A safe, structured error intended for an API client."""

    def __init__(
        self,
        stage: str,
        code: str,
        message: str,
        *,
        status_code: int = 422,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.code = code
        self.message = message
        self.status_code = status_code

    def detail(self, request_id: str) -> dict[str, str]:
        return {
            "request_id": request_id,
            "stage": self.stage,
            "code": self.code,
            "message": self.message,
        }
