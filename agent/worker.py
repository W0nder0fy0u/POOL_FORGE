"""PoolForge Worker Agent Core.

Runs node telemetry monitoring, handles outbound control plane communication with
exponential backoff, executes pipeline partitions, and reports training metrics.
"""

import asyncio
import json
import logging
import platform
import socket
import time
import uuid
from typing import Any, Dict, Optional
import psutil
import requests
import torch
import websockets

from common.datasets import get_dataloader
from common.pipeline import PipelineStage
from common.schemas import AssignJobMessage, NodeTelemetry, StepMetric
from common.transports import DirectTCPTransport, RelayTransport, Transport

logger = logging.getLogger("poolforge.agent")


class AgentWorker:
    """Distributed worker node executing pipeline parallel partitions."""

    def __init__(
        self,
        room_code: str,
        server_url: str = "http://localhost:8000",
        node_id: Optional[str] = None,
        ram_cap_gb: Optional[float] = None,
        transport_type: str = "relay",
        simulate_latency_ms: float = 0.0,
        device: str = "cpu",
    ):
        self.room_code = room_code.upper()
        self.server_url = server_url.rstrip("/")
        self.node_id = node_id or f"node_{uuid.uuid4().hex[:6]}"
        self.ram_cap_gb = ram_cap_gb
        self.transport_type = transport_type.lower()
        self.simulate_latency_ms = simulate_latency_ms
        self.device = torch.device(device)

        self.status = "idle"  # "idle" | "training" | "hosting" | "paused"
        self._running = False
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._active_job_task: Optional[asyncio.Task] = None
        self._current_stage: Optional[PipelineStage] = None
        self._direct_tcp: Optional[DirectTCPTransport] = None
        self._pause_event = asyncio.Event()
        self._pause_event.set()

    def get_telemetry(self) -> NodeTelemetry:
        """Poll host memory, CPU, and platform details."""
        vm = psutil.virtual_memory()
        real_free = vm.available / (1024 ** 3)
        reported_free = (
            min(real_free, self.ram_cap_gb) if self.ram_cap_gb else real_free
        )

        tcp_host = "127.0.0.1"
        tcp_port = self._direct_tcp.listen_port if self._direct_tcp else None

        return NodeTelemetry(
            node_id=self.node_id,
            hostname=socket.gethostname(),
            os_name=platform.system(),
            ram_total_gb=round(vm.total / (1024 ** 3), 2),
            ram_free_gb=round(reported_free, 2),
            cpu_percent=round(psutil.cpu_percent(interval=None), 1),
            status=self.status,
            transport_type=self.transport_type,
            tcp_host=tcp_host,
            tcp_port=tcp_port,
        )

    async def start(self) -> None:
        """Start worker lifecycle: heartbeats and control plane listener with auto-reconnect."""
        self._running = True

        # Pre-initialize DirectTCP transport if specified
        if self.transport_type == "tcp":
            self._direct_tcp = DirectTCPTransport(
                node_id=self.node_id,
                host="127.0.0.1",
                port=0,
                simulate_latency_ms=self.simulate_latency_ms,
            )
            await self._direct_tcp.start()

        # Reconnect loop with exponential backoff (max 10s)
        backoff = 1.0
        while self._running:
            try:
                await self._connect_and_listen()
                backoff = 1.0
            except (websockets.ConnectionClosed, OSError, Exception) as exc:
                if not self._running:
                    break
                logger.warning(
                    f"Connection lost to coordinator ({exc}). Reconnecting in {backoff:.1f}s..."
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 1.5, 10.0)

    async def _connect_and_listen(self) -> None:
        """Establish WebSocket connection and process incoming control messages."""
        ws_url = self.server_url
        if ws_url.startswith("http://"):
            ws_url = "ws://" + ws_url[7:]
        elif ws_url.startswith("https://"):
            ws_url = "wss://" + ws_url[8:]

        endpoint = f"{ws_url}/ws/agent/{self.room_code}/{self.node_id}"
        async with websockets.connect(endpoint) as ws:
            self._ws = ws
            logger.info(f"Agent [{self.node_id}] registered with room {self.room_code}")

            # Send registration
            await ws.send(
                json.dumps(
                    {"type": "register", "telemetry": self.get_telemetry().model_dump()}
                )
            )

            # Spawn heartbeat loop
            heartbeat_task = asyncio.create_task(self._heartbeat_loop(ws))

            try:
                async for message in ws:
                    msg = json.loads(message)
                    mtype = msg.get("type")

                    if mtype == "assign":
                        payload = msg["payload"]
                        assign_msg = AssignJobMessage(**payload)
                        # Cancel existing running job if any
                        if self._active_job_task and not self._active_job_task.done():
                            self._active_job_task.cancel()
                        self._active_job_task = asyncio.create_task(
                            self._execute_job(assign_msg)
                        )

                    elif mtype == "stop":
                        logger.info("Received stop signal from coordinator")
                        if self._active_job_task and not self._active_job_task.done():
                            self._active_job_task.cancel()
                        self.status = "idle"

                    elif mtype == "pause":
                        logger.info("Received pause signal from coordinator")
                        self._pause_event.clear()
                        self.status = "paused"

                    elif mtype == "resume":
                        logger.info("Received resume signal from coordinator")
                        self._pause_event.set()
                        self.status = "training"

            finally:
                heartbeat_task.cancel()

    async def _heartbeat_loop(self, ws: websockets.WebSocketClientProtocol) -> None:
        """Emit telemetry update every 2 seconds."""
        while self._running:
            try:
                await asyncio.sleep(2.0)
                telem = self.get_telemetry()
                await ws.send(
                    json.dumps({"type": "heartbeat", "telemetry": telem.model_dump()})
                )
            except asyncio.CancelledError:
                break
            except Exception:
                break

    async def _execute_job(self, assign: AssignJobMessage) -> None:
        """Execute assigned pipeline partition training."""
        self.status = "training"
        logger.info(
            f"Starting job {assign.job_id}: stage {assign.stage_index}/{assign.num_stages} "
            f"(layers [{assign.start_layer}, {assign.end_layer}))"
        )

        # 1. Setup transport
        transport: Transport
        if self.transport_type == "tcp" and self._direct_tcp:
            transport = self._direct_tcp
            for p_id, (p_host, p_port) in assign.peer_tcp_addresses.items():
                if p_id != self.node_id:
                    self._direct_tcp.register_peer(p_id, p_host, p_port)
        else:
            transport = RelayTransport(
                node_id=self.node_id,
                relay_url=self.server_url,
                room_code=self.room_code,
                simulate_latency_ms=self.simulate_latency_ms,
            )
            await transport.start()

        hp = assign.hyperparameters
        batch_size = hp.get("batch_size", 16)
        microbatches = hp.get("microbatches", 1)
        epochs = hp.get("epochs", 1)
        steps_per_epoch = hp.get("steps_per_epoch", 5)
        lr = hp.get("learning_rate", 0.05)
        optim_type = hp.get("optimizer", "sgd")
        ckpt_interval = hp.get("checkpoint_interval", 5)

        # 2. Build pipeline stage
        stage = PipelineStage(
            stage_index=assign.stage_index,
            num_stages=assign.num_stages,
            layer_specs=assign.layer_specs,
            start_layer=assign.start_layer,
            end_layer=assign.end_layer,
            transport=transport,
            prev_node_id=assign.prev_node_id,
            next_node_id=assign.next_node_id,
            learning_rate=lr,
            optimizer_type=optim_type,
            job_id=assign.job_id,
        )
        self._current_stage = stage

        # 3. First stage loads dataset
        loader = None
        data_iter = None
        if stage.is_first:
            ds_name = hp.get("dataset_name", "synthetic")
            loader = get_dataloader(
                dataset_name=ds_name,
                batch_size=batch_size,
                num_samples=batch_size * steps_per_epoch * epochs,
            )
            data_iter = iter(loader)

        total_steps = epochs * steps_per_epoch

        try:
            for step in range(total_steps):
                # Check for pause before starting step
                await self._pause_event.wait()
                t_start = time.perf_counter()
                stage.zero_grad()

                x_batch, y_batch = None, None
                if stage.is_first and data_iter:
                    try:
                        x_batch, y_batch = next(data_iter)
                    except StopIteration:
                        data_iter = iter(loader)
                        x_batch, y_batch = next(data_iter)

                # Microbatch execution
                m_inputs = (
                    torch.chunk(x_batch, microbatches)
                    if x_batch is not None
                    else [None] * microbatches
                )
                m_labels = (
                    torch.chunk(y_batch, microbatches)
                    if y_batch is not None
                    else [None] * microbatches
                )

                step_loss = 0.0

                # Forward microbatches
                for m_idx in range(microbatches):
                    loss_m = await stage.forward_step(
                        x=m_inputs[m_idx],
                        labels=m_labels[m_idx],
                        step=step,
                        microbatch=m_idx,
                    )
                    if loss_m is not None:
                        step_loss += loss_m / microbatches

                # Backward microbatches
                for m_idx in range(microbatches):
                    await stage.backward_step(
                        step=step,
                        microbatch=m_idx,
                        loss_scale=1.0 / microbatches,
                    )

                stage.step_optimizer()
                step_time_ms = (time.perf_counter() - t_start) * 1000.0
                throughput = (
                    batch_size / (step_time_ms / 1000.0) if step_time_ms > 0 else 0.0
                )

                # Checkpoint
                if (step + 1) % ckpt_interval == 0:
                    stage.save_checkpoint(step + 1)

                # Last stage reports metric to Coordinator
                if stage.is_last:
                    metric = StepMetric(
                        job_id=assign.job_id,
                        step=step + 1,
                        epoch=(step // steps_per_epoch) + 1,
                        loss=round(step_loss, 4),
                        step_time_ms=round(step_time_ms, 2),
                        throughput_samples_sec=round(throughput, 2),
                        network_latency_ms=self.simulate_latency_ms,
                    )
                    try:
                        requests.post(
                            f"{self.server_url}/api/rooms/{self.room_code}/report_metric",
                            json=metric.model_dump(),
                            timeout=2.0,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to post metric: {e}")

                logger.info(
                    f"Step {step+1}/{total_steps} completed in {step_time_ms:.1f}ms "
                    + (f"(loss: {step_loss:.4f})" if stage.is_last else "")
                )

        except asyncio.CancelledError:
            logger.info(f"Job {assign.job_id} cancelled")
        except Exception as e:
            logger.error(f"Error during job execution: {e}", exc_info=True)
        finally:
            if self.transport_type != "tcp":
                await transport.stop()
            self.status = "idle"
            logger.info(f"Job {assign.job_id} finished")

    async def stop(self) -> None:
        """Gracefully terminate worker."""
        self._running = False
        if self._active_job_task and not self._active_job_task.done():
            self._active_job_task.cancel()
        if self._direct_tcp:
            await self._direct_tcp.stop()
        if self._ws:
            await self._ws.close()
