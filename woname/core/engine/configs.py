from pydantic import BaseModel
from typing import Optional, Literal
from woname.evaluators.configs import MetricConfig

class TrainerConfig(BaseModel):
    epochs: int = 10
    device: Literal[
                    "cpu",
                    "cuda",
                    "mps",
                    ] = "cpu"
    log_evey: int = 10
    evaluators: Optional[
        list[MetricConfig]
    ] = None
