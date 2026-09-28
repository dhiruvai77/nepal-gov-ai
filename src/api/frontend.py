"""Static web-interface registration for NepalGov AI.

The frontend is served by the same FastAPI process as the application API.
Keeping the browser client same-origin avoids a separate frontend service and
CORS configuration for the initial application release.
"""

from __future__ import annotations

from pathlib import (
    Path,
)

from fastapi import (
    FastAPI,
)
from fastapi.responses import (
    FileResponse,
)
from fastapi.staticfiles import (
    StaticFiles,
)


WEB_ROOT = (
    Path(
        __file__
    )
    .resolve()
    .parents[
        1
    ]
    / "web"
)

INDEX_PATH = (
    WEB_ROOT
    / "index.html"
)

INDEX_HEADERS = {
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": (
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self'; "
        "connect-src 'self'; "
        "img-src 'self' data:; "
        "font-src 'self'; "
        "object-src 'none'; "
        "base-uri 'none'; "
        "frame-ancestors 'none'; "
        "form-action 'self'"
    ),
}


def register_frontend(
    application: FastAPI,
) -> None:
    """Attach the static browser application to one FastAPI instance."""

    if not isinstance(
        application,
        FastAPI,
    ):
        raise TypeError(
            "application must be a FastAPI instance."
        )

    if not INDEX_PATH.is_file():
        raise RuntimeError(
            "NepalGov AI web index does not exist: "
            f"{INDEX_PATH}"
        )

    application.mount(
        "/static",
        StaticFiles(
            directory=str(
                WEB_ROOT
            ),
        ),
        name="static",
    )

    @application.get(
        "/",
        include_in_schema=False,
    )
    def web_index(
    ) -> FileResponse:
        """Serve the browser application entry point."""

        return FileResponse(
            path=INDEX_PATH,
            media_type="text/html",
            headers=INDEX_HEADERS,
        )