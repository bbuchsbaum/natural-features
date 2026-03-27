"""Slurm directive mapping helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SlurmResources:
    cpus: int = 1
    mem_gb: int = 4
    time_min: int = 30
    gpus: int = 0
    partition: str | None = None
    qos: str | None = None
    account: str | None = None


def to_sbatch_args(resources: SlurmResources) -> list[str]:
    args = [
        f"--cpus-per-task={resources.cpus}",
        f"--mem={resources.mem_gb}G",
        f"--time={resources.time_min}",
    ]
    if resources.gpus > 0:
        args.append(f"--gpus={resources.gpus}")
    if resources.partition:
        args.append(f"--partition={resources.partition}")
    if resources.qos:
        args.append(f"--qos={resources.qos}")
    if resources.account:
        args.append(f"--account={resources.account}")
    return args


def profile_to_resources(profile: str) -> SlurmResources:
    key = profile.strip().lower()
    if key == "short":
        return SlurmResources(cpus=2, mem_gb=8, time_min=30)
    if key == "long":
        return SlurmResources(cpus=4, mem_gb=16, time_min=240)
    if key == "gpu":
        return SlurmResources(cpus=4, mem_gb=32, time_min=120, gpus=1)
    return SlurmResources()

