from pydantic import BaseModel

class TrainerConfig(BaseModel):
    epochs: int = 10
    device: str = "cpu"
    log_evey: int = 10