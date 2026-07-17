"""Protocol definitions shared by Physics-IQ Original and Verified."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

PhysicsIQProtocol = Literal["original", "verified"]


@dataclass(frozen=True)
class PhysicsIQProtocolSpec:
    """Immutable differences between the two official benchmark protocols."""

    protocol: PhysicsIQProtocol
    benchmark_id: str
    display_name: str
    dataset_dir_name: str
    prompt_asset: Path
    primary_score_key: str
    primary_metric_id: str


ORIGINAL = PhysicsIQProtocolSpec(
    protocol="original",
    benchmark_id="physics-iq",
    display_name="Physics-IQ Original",
    dataset_dir_name="physics-IQ-benchmark",
    prompt_asset=Path("descriptions/descriptions_original.csv"),
    primary_score_key="final_score_orig",
    primary_metric_id="physics_iq_score",
)

VERIFIED = PhysicsIQProtocolSpec(
    protocol="verified",
    benchmark_id="physics-iq-verified",
    display_name="Physics-IQ Verified",
    dataset_dir_name="physics-IQ-benchmark-verified",
    prompt_asset=Path("descriptions/descriptions_base.csv"),
    primary_score_key="final_score_view",
    primary_metric_id="physics_iq_verified_score",
)

PROTOCOLS: dict[PhysicsIQProtocol, PhysicsIQProtocolSpec] = {
    "original": ORIGINAL,
    "verified": VERIFIED,
}
PROTOCOLS_BY_BENCHMARK_ID = {spec.benchmark_id: spec for spec in PROTOCOLS.values()}


def resolve_protocol(
    *,
    benchmark_id: str | None = None,
    protocol: str | None = None,
) -> PhysicsIQProtocolSpec:
    """Resolve a protocol and reject contradictory CLI/config choices."""

    by_id = PROTOCOLS_BY_BENCHMARK_ID.get(benchmark_id or "")
    if protocol is None:
        return by_id or ORIGINAL
    normalized = protocol.strip().lower()
    if normalized not in PROTOCOLS:
        raise ValueError(f"Unknown Physics-IQ protocol {protocol!r}; use 'original' or 'verified'.")
    explicit = PROTOCOLS[normalized]  # type: ignore[index]
    if by_id is not None and by_id.protocol != explicit.protocol:
        raise ValueError(
            f"Benchmark {benchmark_id!r} selects {by_id.protocol!r}, but --protocol selects "
            f"{explicit.protocol!r}."
        )
    return explicit
