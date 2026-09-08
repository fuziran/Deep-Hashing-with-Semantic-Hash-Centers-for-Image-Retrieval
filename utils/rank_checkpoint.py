"""Epoch-boundary SBR checkpoints. Load only trusted local training artifacts."""
import os
import random

import numpy as np
import torch

VERSION = "sbr-epoch-checkpoint-v1"


def atomic_save(value, path):
    temporary = str(path) + ".tmp"
    torch.save(value, temporary)
    os.replace(temporary, path)


def save_checkpoint(path, args, net, optimizer, scheduler, ranker, loader,
                    epoch, global_step, best_state, best_epoch, best_map, history):
    if getattr(loader, "persistent_workers", False):
        raise ValueError("Exact epoch resume requires non-persistent data workers")
    state = dict(version=VERSION, run_id=args.run_id, epoch=epoch, global_step=global_step,
                 model=net.state_dict(), optimizer=optimizer.state_dict(),
                 scheduler=scheduler.state_dict(), ranker=ranker.state_dict(),
                 best_state=best_state, best_epoch=best_epoch, best_map=best_map, history=history,
                 python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                 torch_rng=torch.get_rng_state(),
                 cuda_rng=torch.cuda.get_rng_state_all() if args.device.type == "cuda" else None,
                 device_type=args.device.type,
                 loader_rng=loader.generator.get_state() if loader.generator is not None else None)
    atomic_save(state, path)


def restore_checkpoint(path, args, net, optimizer, scheduler, ranker, loader):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("version") != VERSION or state.get("run_id") != args.run_id:
        raise ValueError("Resume requires identical source, parent artifacts and run configuration")
    if state.get("device_type") != args.device.type:
        raise ValueError("Resume device type mismatch")
    if (not isinstance(state["epoch"], int) or not 0 < state["epoch"] <= args.epoch
            or state["global_step"] != state["epoch"] * len(loader)
            or len(state["history"]) != state["epoch"]):
        raise ValueError("Invalid completed epoch in checkpoint")
    if getattr(loader, "persistent_workers", False):
        raise ValueError("Exact epoch resume requires non-persistent data workers")
    if (state["loader_rng"] is None) != (loader.generator is None):
        raise ValueError("Resume data-loader generator mismatch")
    ranker.load_state_dict(state["ranker"])
    net.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"] is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    if loader.generator is not None:
        loader.generator.set_state(state["loader_rng"])
    return state
