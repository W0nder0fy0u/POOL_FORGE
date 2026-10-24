"""Pydantic schemas for every control-plane message and REST body.

Every message that crosses the network is parsed with one of these models,
so malformed or unexpected data is rejected before any code acts on it.
(Tensors don't go through here: they use the binary format in protocol.py.)
"""

from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

ROOM_CODE_PATTERN = r"^[A-Z0-9]{6}$"
NODE_ID_PATTERN = r"^[A-Za-z0-9_.-]{1,40}$"


class Strict(BaseModel):
    """Base: unknown fields are an error, not silently ignored."""

    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------- REST
class JoinRequest(Strict):
    hostname: str = Field(max_length=100)
    os: str = Field(max_length=100)
    node_id: Optional[str] = Field(default=None, pattern=NODE_ID_PATTERN)


class JoinResponse(Strict):
    node_id: str
    node_token: str
    tcp_secret: str
    heartbeat_s: float


class TrainingConfig(Strict):
    """What the admin picks on the Training page."""

    model: str = Field(default="mnist_cnn", max_length=40)
    model_options: Dict[str, int] = Field(default_factory=dict)  # e.g. {"width": 2048, "depth": 8}
    dataset: Optional[Literal["synthetic", "mnist"]] = None  # None = model's default dataset
    batch_size: int = Field(default=64, ge=1, le=4096)
    microbatches: int = Field(default=1, ge=1, le=64)
    epochs: int = Field(default=1, ge=1, le=1000)
    steps_per_epoch: int = Field(default=100, ge=1, le=100000)
    lr: float = Field(default=1e-3, gt=0, le=10)
    optimizer: Literal["adam", "sgd"] = "adam"
    checkpoint_every: int = Field(default=20, ge=1, le=100000)
    seed: int = Field(default=0, ge=0, le=2**31)


class DeployRequest(Strict):
    app_id: str = Field(max_length=40)
    node_id: Optional[str] = Field(default=None, pattern=NODE_ID_PATTERN)


class LockRequest(Strict):
    locked: bool


# ------------------------------------------------- agent -> coordinator (WS)
class StatsMsg(Strict):
    type: Literal["stats"]
    hostname: str = Field(max_length=100)
    os: str = Field(max_length=100)
    ram_total_gb: float = Field(ge=0)
    ram_free_gb: float = Field(ge=0)  # already min(real free, --ram-cap-gb)
    ram_cap_gb: Optional[float] = Field(default=None, ge=0)
    proc_ram_mb: float = Field(ge=0)  # RAM used by the agent process itself
    cpu_percent: float = Field(ge=0, le=100 * 1024)
    status: Literal["idle", "training", "paused", "hosting"]
    transport: Literal["relay", "tcp"]
    tcp_host: Optional[str] = Field(default=None, max_length=100)
    tcp_port: Optional[int] = Field(default=None, ge=1, le=65535)
    docker: bool = False
    cloudflared: bool = False
    # GPU sharing (optional, so older agents still validate)
    device: Literal["cpu", "cuda", "mps"] = "cpu"
    gpu_name: Optional[str] = Field(default=None, max_length=100)
    vram_total_gb: Optional[float] = Field(default=None, ge=0)
    vram_free_gb: Optional[float] = Field(default=None, ge=0)  # already min(real free, --vram-cap-gb)
    vram_cap_gb: Optional[float] = Field(default=None, ge=0)


class StepMsg(Strict):
    type: Literal["step"]
    job_id: str = Field(max_length=80)
    stage: int = Field(ge=0)
    step: int = Field(ge=1)
    epoch: int = Field(ge=1)
    loss: Optional[float] = None  # only from the last stage
    step_time_ms: float = Field(ge=0)
    bytes_sent: int = Field(ge=0)
    bytes_received: int = Field(ge=0)
    avg_send_ms: float = Field(ge=0)
    proc_ram_mb: float = Field(ge=0)
    gpu_mem_mb: Optional[float] = Field(default=None, ge=0)  # peak VRAM used by this stage (GPU nodes only)


class CheckpointMsg(Strict):
    type: Literal["checkpoint"]
    job_id: str = Field(max_length=80)
    stage: int = Field(ge=0)
    step: int = Field(ge=0)


class JobStateMsg(Strict):
    type: Literal["job_state"]
    job_id: str = Field(max_length=80)
    stage: int = Field(ge=0)
    state: Literal["started", "done", "error"]
    detail: str = Field(default="", max_length=2000)


class AppStateMsg(Strict):
    type: Literal["app_state"]
    deployment_id: str = Field(max_length=40)
    status: Literal["starting", "running", "stopped", "error"]
    url: Optional[str] = Field(default=None, max_length=300)
    detail: str = Field(default="", max_length=2000)


AgentMessage = Annotated[
    Union[StatsMsg, StepMsg, CheckpointMsg, JobStateMsg, AppStateMsg], Field(discriminator="type")
]
agent_message_adapter = TypeAdapter(AgentMessage)


# ------------------------------------------------- coordinator -> agent (WS)
class HyperParams(Strict):
    batch_size: int
    microbatches: int
    epochs: int
    steps_per_epoch: int
    lr: float
    optimizer: str
    checkpoint_every: int
    seed: int
    dataset: str


class AssignMsg(Strict):
    type: Literal["assign"]
    job_id: str
    stage_index: int
    num_stages: int
    start_layer: int
    end_layer: int
    specs: List[Dict[str, Any]]
    input_shape: List[int]
    num_classes: int
    prev_node: Optional[str]
    next_node: Optional[str]
    transport: Literal["relay", "tcp"]
    peers: Dict[str, List[Any]] = Field(default_factory=dict)  # node_id -> [host, port] (tcp only)
    hp: HyperParams
    resume_step: int = 0  # >0: load this checkpoint first, then continue
    device: Literal["cpu", "cuda", "mps"] = "cpu"  # where the planner wants this stage to run
    attempt: int = 0  # bumped on every re-assign, so stale frames of older attempts are ignored


class ControlMsg(Strict):
    type: Literal["control"]
    action: Literal["pause", "resume", "stop"]
    job_id: str
    reason: str = ""


class RunAppMsg(Strict):
    type: Literal["run_app"]
    deployment_id: str
    app_id: str


class StopAppMsg(Strict):
    type: Literal["stop_app"]
    deployment_id: str


CoordinatorMessage = Annotated[Union[AssignMsg, ControlMsg, RunAppMsg, StopAppMsg], Field(discriminator="type")]
coordinator_message_adapter = TypeAdapter(CoordinatorMessage)


# ------------------------------------------- browser -> coordinator (WS)
# Chat and voice-call signalling from the dashboard. Audio itself goes browser
# to browser (WebRTC); the coordinator only forwards these small messages.
PEER_ID_PATTERN = r"^[0-9a-f]{8}$"


class ChatIn(Strict):
    type: Literal["chat"]
    name: str = Field(min_length=1, max_length=30)
    text: str = Field(min_length=1, max_length=500)


class CallJoin(Strict):
    type: Literal["call_join"]
    name: str = Field(min_length=1, max_length=30)


class CallLeave(Strict):
    type: Literal["call_leave"]


class Signal(Strict):
    type: Literal["signal"]
    to: str = Field(pattern=PEER_ID_PATTERN)
    data: Dict[str, Any]  # WebRTC offer / answer / ICE candidate, passed through as-is


BrowserMessage = Annotated[Union[ChatIn, CallJoin, CallLeave, Signal], Field(discriminator="type")]
browser_message_adapter = TypeAdapter(BrowserMessage)
