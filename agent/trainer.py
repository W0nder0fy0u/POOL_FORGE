"""Runs one assigned pipeline stage on this laptop.

Called by the agent when the coordinator sends an `assign` message. Builds
ONLY this node's layers, (first stage) loads data locally, trains step by
step, saves checkpoints every K steps and reports metrics back.
"""

import asyncio
import logging
import time
from typing import Awaitable, Callable, Optional

import psutil

from common.datasets import get_batches
from common.pipeline import PipelineStage
from common.schemas import AssignMsg, CheckpointMsg, JobStateMsg, StepMsg
from common.transports import Transport

logger = logging.getLogger("poolforge.trainer")

Report = Callable[[object], Awaitable[None]]


def proc_ram_mb() -> float:
    """RAM used by this agent process (weights, activations, PyTorch itself)."""
    return psutil.Process().memory_info().rss / (1024 * 1024)


async def run_job(
    assign: AssignMsg,
    transport: Transport,
    report: Report,
    gate: asyncio.Event,
    encrypt_key: Optional[bytes] = None,
    checkpoint_dir: str = "./checkpoints",
    data_dir: str = "./data",
) -> None:
    """Train this node's stage until all steps are done (or we are cancelled).

    `gate` is set while the job may run and cleared while it is paused; we
    only check it between steps, so a step is never cut in half.
    """
    hp = assign.hp
    i = assign.stage_index
    try:
        stage = PipelineStage(
            assign.specs, assign.start_layer, assign.end_layer, i, assign.num_stages, transport,
            assign.prev_node, assign.next_node, assign.job_id, lr=hp.lr, optimizer=hp.optimizer,
            seed=hp.seed, encrypt_key=encrypt_key, checkpoint_dir=checkpoint_dir, attempt=assign.attempt,
        )
        start = 0
        if assign.resume_step > 0:
            if not stage.load_checkpoint(assign.resume_step):
                raise RuntimeError(f"checkpoint for step {assign.resume_step} not found on this node")
            start = assign.resume_step
            logger.info("Resumed from checkpoint at step %d", start)

        detail = ""
        batches = None
        if stage.is_first:  # only the first stage needs the dataset
            batches, detail = await asyncio.to_thread(
                get_batches, hp.dataset, assign.input_shape, assign.num_classes, hp.batch_size, hp.seed, data_dir
            )
            for _ in range(start):  # skip batches already trained on, to stay deterministic
                next(batches)
        logger.info(
            "Job %s: stage %d/%d, layers [%d, %d), %s params%s",
            assign.job_id, i + 1, assign.num_stages, assign.start_layer, assign.end_layer,
            f"{stage.num_params():,}", f", data: {detail}" if detail else "",
        )
        await report(JobStateMsg(type="job_state", job_id=assign.job_id, stage=i, state="started", detail=detail))
        transport.take_step_stats()  # reset counters

        total = hp.epochs * hp.steps_per_epoch
        for step in range(start, total):
            await gate.wait()  # blocks here while paused
            t0 = time.perf_counter()
            x = y = None
            if stage.is_first:
                x, y = next(batches)
            loss = await stage.train_step(step, x, y, hp.microbatches)
            step_ms = (time.perf_counter() - t0) * 1000.0
            net = transport.take_step_stats()
            await report(StepMsg(
                type="step", job_id=assign.job_id, stage=i, step=step + 1, epoch=step // hp.steps_per_epoch + 1,
                loss=loss, step_time_ms=step_ms, bytes_sent=net["bytes_sent"], bytes_received=net["bytes_received"],
                avg_send_ms=net["avg_send_ms"], proc_ram_mb=proc_ram_mb(),
            ))
            if loss is not None and (step + 1) % 10 == 0:
                logger.info("step %d/%d  loss %.4f  %.0f ms/step", step + 1, total, loss, step_ms)
            if (step + 1) % hp.checkpoint_every == 0 or step + 1 == total:
                await asyncio.to_thread(stage.save_checkpoint, step + 1)
                await report(CheckpointMsg(type="checkpoint", job_id=assign.job_id, stage=i, step=step + 1))

        await report(JobStateMsg(type="job_state", job_id=assign.job_id, stage=i, state="done"))
        logger.info("Job %s finished on this node", assign.job_id)
    except asyncio.CancelledError:
        logger.info("Job %s stopped on this node", assign.job_id)
        raise
    except Exception as exc:
        logger.exception("Job %s failed", assign.job_id)
        await report(JobStateMsg(type="job_state", job_id=assign.job_id, stage=i, state="error", detail=str(exc)[:1900]))
