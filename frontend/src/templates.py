"""Shared Jinja2Templates instance.

Centralises the templates directory path so all routes use the same
configuration. Templates are resolved relative to the working directory
(frontend/).
"""

import os

from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="templates")


def _stylesheet_version() -> str:
    """Return a token that changes whenever the stylesheet changes.

    Browsers cache /static/style.css aggressively, so after a deploy a page
    can render with the previous rules and look broken for reasons that are
    not in the code. Appending the file's modification time to the URL makes
    the browser fetch the new file, and only when there is a new file.
    """
    try:
        return str(int(os.path.getmtime("static/style.css")))
    except OSError:
        return "0"


templates.env.globals["stylesheet_version"] = _stylesheet_version()
