"""PoolForge Shared Message Schemas.

Pydantic models for strict network validation across the control plane,
REST APIs, and dashboard WebSockets.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class NodeTelemetry(BaseModel):
    """Hardware telemetry and operational status reported by worker agents."""
    node_id: str
    hostname: str
    os_name: str
    ram_total_gb: float
    ram_free_gb: float
    cpu_percent: float
    status: str = "idle"  # "idle" | "training" | "hosting" | "offline"
    transport_type: str = "relay"
    tcp_host: Optional[str] = None
    tcp_port: Optional[int] = None
    assigned_layers: Optional[List[int]] = None  # [start, end]


class CreateRoomResponse(BaseModel):
    """Credentials returned upon creating a new collaborative pool."""
    room_code: str
    admin_token: str
    join_url: str
    created_at: float


class RoomStatus(BaseModel):
    """Live room state including participating nodes and current job status."""
    room_code: str
    node_count: int
    nodes: List[NodeTelemetry]
    active_job_id: Optional[str] = None
    job_status: str = "idle"  # "idle" | "running" | "paused" | "stopped" | "completed" | "error"


class SplitPlanStage(BaseModel):
    """Scheduler allocation for an individual node stage."""
    node_id: str
    stage_index: int
    start_layer: int
    end_layer: int
    num_layers: int
    estimated_ram_mb: float
    node_ram_cap_gb: float


class StartTrainingRequest(BaseModel):
    """Admin request to trigger distributed pipeline training."""
    model_name: str = "tiny_test"
    dataset_name: str = "synthetic"
    batch_size: int = 16
    microbatches: int = 1
    epochs: int = 1
    steps_per_epoch: int = 10
    learning_rate: float = 0.05
    optimizer: str = "sgd"
    checkpoint_interval: int = 5


class AssignJobMessage(BaseModel):
    """Job assignment payload sent from Coordinator to worker Agent."""
    job_id: str
    stage_index: int
    num_stages: int
    start_layer: int
    end_layer: int
    layer_specs: List[Dict[str, Any]]
    prev_node_id: Optional[str] = None
    next_node_id: Optional[str] = None
    transport_type: str = "relay"
    peer_tcp_addresses: Dict[str, List[Any]] = Field(default_factory=dict)
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)


class StepMetric(BaseModel):
    """Telemetry emitted per training step."""
    job_id: str
    step: int
    epoch: int
    loss: float
    step_time_ms: float
    throughput_samples_sec: float
    per_node_ram_mb: Dict[str, float] = Field(default_factory=dict)
    per_node_cpu_percent: Dict[str, float] = Field(default_factory=dict)
    network_latency_ms: float = 0.0
    bytes_sent: int = 0
