"""Gold standard URL and domain queries.

Supported domains are derived from the ``gold_standard`` / ``web_resources``
tables in MariaDB: a domain is "supported" as soon as at least one gold
standard entry exists for it. The tables are seeded from ``gs_data/*_gs.json``
at first boot by ``db.init_loader``; new domains added at runtime via
``/add_web_resource`` + ``/add_gold_standard`` become supported immediately.
"""

from ..db import queries


def get_domains() -> list[str]:
    """Return every domain that has at least one gold standard entry."""
    return queries.get_distinct_gold_standard_domains()


def is_supported_domain(domain: str) -> bool:
    """Return True if the domain has at least one gold standard entry."""
    return domain in get_domains()


def get_urls_for_domain(domain: str) -> list[str]:
    """Return all GS URLs belonging to the given domain."""
    return queries.get_urls_by_domain(domain)
