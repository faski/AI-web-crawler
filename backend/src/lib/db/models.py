"""Pydantic models for rows read from the database.

Read queries in ``queries.py`` return these classes instead of plain dicts.
Benefits:
  - attribute access (``resource.url`` instead of ``resource["url"]``)
  - automatic type checks (Pydantic raises an error if a field is missing)
  - easy conversion to a plain dict via ``.model_dump()`` when needed by a
    route handler
"""

from pydantic import BaseModel


class WebResource(BaseModel):
    url: str
    domain: str
    title: str = ""
    html_text: str


class GoldStandardEntry(BaseModel):
    url: str
    domain: str
    title: str = ""
    html_text: str
    gold_text: str
