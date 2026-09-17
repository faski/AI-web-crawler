"""Shared Jinja2Templates instance.

Centralises the templates directory path so all routes use the same
configuration. Templates are resolved relative to the working directory
(frontend/).
"""

import os

from fastapi.templating import Jinja2Templates
from jinja2 import select_autoescape

templates = Jinja2Templates(directory="templates")

# Jinja decides whether to escape by looking at the file extension, and its
# default list is ("html", "htm", "xml"). Every template here is named
# *.html.jinja, so the extension it sees is "jinja" and it escaped nothing:
# any value containing a '<' went into the page as markup.
#
# That stayed invisible while the data was URLs and numbers. It surfaced when
# the self-check notes arrived - prose written by a model about HTML, so full
# of real tags. One note containing <style> turned the rest of the table into
# a stylesheet and fifteen rows vanished from the page while being present in
# the response.
templates.env.autoescape = select_autoescape(
    enabled_extensions=("html", "htm", "xml", "jinja"),
    default_for_string=True,
)


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
