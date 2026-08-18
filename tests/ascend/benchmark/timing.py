import math
import statistics
from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class TimingSummary:
    minimum: float
    mean: float
    p50: float
    p95: float
    maximum: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _percentile(sorted_samples: tuple[float, ...], quantile: float) -> float:
    position = (len(sorted_samples) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_samples[lower]
    fraction = position - lower
    return (
        sorted_samples[lower] * (1.0 - fraction)
        + sorted_samples[upper] * fraction
    )


def summarize_samples(samples: Iterable[float]) -> TimingSummary:
    values = tuple(float(sample) for sample in samples)
    if not values or any(
        not math.isfinite(sample) or sample <= 0.0 for sample in values
    ):
        raise ValueError("timing samples must be finite and positive")
    ordered = tuple(sorted(values))
    return TimingSummary(
        minimum=ordered[0],
        mean=statistics.fmean(ordered),
        p50=_percentile(ordered, 0.50),
        p95=_percentile(ordered, 0.95),
        maximum=ordered[-1],
    )


def logical_gbps(num_bytes: int, seconds: float) -> float:
    if num_bytes < 0:
        raise ValueError("logical bytes must be nonnegative")
    if not math.isfinite(seconds) or seconds <= 0.0:
        raise ValueError("elapsed seconds must be finite and positive")
    return num_bytes / seconds / 1e9
