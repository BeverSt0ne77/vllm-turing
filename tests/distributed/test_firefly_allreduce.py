# SPDX-License-Identifier: Apache-2.0
"""Two-process validation of FireflyAllReduce (fp8 allreduce, SM75).

Spawns TP2 ranks, runs the firefly allreduce on random fp16 tensors via both
the P2P (NVLink IPC) and SHM backends, and compares against
torch.distributed.all_reduce.

Run: `python tests/distributed/test_firefly_allreduce.py`
"""

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

MASTER_PORT = 29711


def _rank_worker(rank: int, world_size: int, backend_env: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(MASTER_PORT)
    os.environ["VLLM_FIREFLY_AR"] = "fp8"
    os.environ["VLLM_FIREFLY_AR_BACKEND"] = backend_env
    os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{rank}")

    from vllm.distributed.device_communicators.firefly_allreduce import (
        FireflyAllReduce,
    )

    shm_name = f"firefly_ar_test_{backend_env}"
    ar = FireflyAllReduce(
        rank_in_group=rank,
        world_size=world_size,
        device=device,
        shm_name=shm_name,
        group=dist.group.WORLD,
    )
    if rank == 0:
        assert not ar.disabled, "firefly AR failed to initialize"
        print(f"backend={ar._backend} initialized")

    torch.manual_seed(1234 + rank)
    for trial, size in enumerate([512, 8192, 1 << 20]):
        x = torch.randn(size, dtype=torch.float16, device=device)

        ref = x.clone()
        dist.all_reduce(ref)

        out = ar.all_reduce(x)
        torch.cuda.synchronize()

        if rank == 0:
            # Per-element relative error explodes on elements where ref ~ 0,
            # which fp8 quantization cannot preserve. Judge the whole tensor
            # instead: max abs error relative to the reference magnitude.
            abs_err = (out - ref).abs().max().item()
            scale = ref.abs().max().item()
            rel = abs_err / max(scale, 1e-6)
            print(
                f"[{backend_env}] n={size}: max_abs_err={abs_err:.4f} "
                f"rel_to_amax={rel:.4f} finite={torch.isfinite(out).all().item()}",
                flush=True,
            )
            assert torch.isfinite(out).all(), "non-finite output"
            # fp8 e4m3 per-token amax: worst-case element lands mid-scale,
            # expect a few percent of the tensor max.
            assert rel < 0.05, f"error too large: rel_to_amax={rel}"

    # Repeated calls to exercise the sequence-counter protocol.
    for _ in range(16):
        x = torch.randn(4096, dtype=torch.float16, device=device)
        ref = x.clone()
        dist.all_reduce(ref)
        out = ar.all_reduce(x)
        torch.cuda.synchronize()
        if rank == 0:
            abs_err = (out - ref).abs().max().item()
            rel = abs_err / max(ref.abs().max().item(), 1e-6)
            assert rel < 0.05, f"steady-state error too large: {rel}"
    if rank == 0:
        print(f"[{backend_env}] steady-state 16 rounds OK", flush=True)

    ar.destroy()
    dist.destroy_process_group()
    if rank == 0:
        print(f"[{backend_env}] PASS")


def main() -> None:
    # Note: the SHM backend cannot initialize on this host (cudaHostRegister
    # fails with err=304, "operation not supported on this OS"); it targets
    # PHB/PCIe hosts without P2P (e.g. T10) and is out of scope here. The
    # NVLink P2P backend is the relevant path for this machine.
    world_size = 2
    import sys

    backends = sys.argv[1:] or ["p2p"]
    for backend in backends:
        mp.spawn(
            _rank_worker,
            args=(world_size, backend),
            nprocs=world_size,
            join=True,
        )
    print("ALL PASS")


if __name__ == "__main__":
    main()
