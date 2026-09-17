"""What it costs to run a pipeline on this machine, in euros.

Crawl4AI is free software, so its price is zero and the interesting question
is a different one: what does running it actually cost the person running it.
That is electricity plus the share of the computer worn out while it ran - and
on these numbers the second is the larger of the two.

This puts the two pipelines on one axis. It does not make them symmetric, and
the difference has to be stated wherever the figure is shown: what OpenRouter
bills includes their hardware, their electricity **and their margin**, while
what is computed here has no margin in it, because nobody is being paid. The
honest label is "what it costs whoever runs it", not "what it is worth".

Every parameter is read from the environment rather than written into the
code, because none of them is a measurement of this project: they are facts
about one laptop and one electricity contract, and someone running this
elsewhere has different ones.

    LOCAL_POWER_WATTS         draw while working, above idle (default 1.52,
                              measured on this MacBook Air M1 with
                              powermetrics: 1.65 W busy, 0.13 W idle)
    ELECTRICITY_EUR_PER_KWH   what a kilowatt-hour costs (default 0.236,
                              from this user's bill)
    HARDWARE_EUR              what the machine cost (default 1250.74, the
                              price of this MacBook Air M1 in 2020)
    HARDWARE_LIFETIME_YEARS   over how long it is written off (default 8).
                              Not four: the machine is six years old and
                              still running this project, so four would be
                              a lifetime it has already outlived, and would
                              overstate every second of it by half.
"""

import os
from dataclasses import dataclass

DEFAULT_POWER_WATTS = 1.52
DEFAULT_EUR_PER_KWH = 0.236
DEFAULT_HARDWARE_EUR = 1250.74
DEFAULT_HARDWARE_LIFETIME_YEARS = 8.0

SECONDS_PER_YEAR = 365 * 24 * 3600


@dataclass(frozen=True)
class LocalCost:
    """What a stretch of local computation cost, split by where it went."""

    seconds: float
    electricity_eur: float
    hardware_eur: float

    @property
    def total_eur(self) -> float:
        """Return the whole cost of the run."""
        return self.electricity_eur + self.hardware_eur


def parameters() -> dict:
    """Return the four numbers the estimate rests on.

    Returned rather than hidden so a page showing a cost can also show what it
    assumed: an estimate whose assumptions are not visible reads as a
    measurement, which this is not.
    """
    return {
        "power_watts": float(
            os.environ.get("LOCAL_POWER_WATTS", DEFAULT_POWER_WATTS)
        ),
        "eur_per_kwh": float(
            os.environ.get("ELECTRICITY_EUR_PER_KWH", DEFAULT_EUR_PER_KWH)
        ),
        "hardware_eur": float(
            os.environ.get("HARDWARE_EUR", DEFAULT_HARDWARE_EUR)
        ),
        "hardware_lifetime_years": float(
            os.environ.get("HARDWARE_LIFETIME_YEARS", DEFAULT_HARDWARE_LIFETIME_YEARS)
        ),
    }


REMOTE_PROVIDERS = frozenset({"openrouter"})
"""Providers whose work happens on somebody else's hardware.

A run against one of these spends its wall-clock time waiting on a network,
not computing: the laptop is close to idle throughout. Charging it for the
power it would draw while working would invent an expense nobody paid, and -
worse - would put a figure in the same column as a real invoice.
"""


def local_cost(seconds: float | None, provider: str | None = None) -> LocalCost | None:
    """Return what ``seconds`` of work on this machine cost, or None.

    None when the work did not happen on this machine, and None for a run with
    no recorded duration. Both cases mean "there is no local cost to report",
    which a caller must render as an absence rather than as a zero.

    Wall-clock seconds, not CPU seconds. The machine draws power for as long
    as the work takes, whether or not every core is busy, and wears out for
    exactly as long; CPU time would credit a pipeline for being
    single-threaded. The seconds are summed per page, so work done in parallel
    is counted once per page rather than once per elapsed second - which is
    the right way round for energy, since running four pages at once draws
    more power than running one.
    """
    if provider is not None and provider.strip().lower() in REMOTE_PROVIDERS:
        return None
    if not seconds or seconds <= 0:
        return None
    values = parameters()
    electricity = (
        values["power_watts"] * seconds / 3600 / 1000 * values["eur_per_kwh"]
    )
    hardware = (
        values["hardware_eur"]
        / (values["hardware_lifetime_years"] * SECONDS_PER_YEAR)
        * seconds
    )
    return LocalCost(
        seconds=seconds, electricity_eur=electricity, hardware_eur=hardware
    )
