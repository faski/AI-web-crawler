"""Shared Jinja2Templates instance.

Centralises the templates directory path so all routes use the same configuration.
Templates are resolved relative to the working directory (frontend/).
"""

from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="templates")
