import onnxruntime
from typing import Dict, Optional

from loguru import logger
from pydantic import BaseModel
import numpy as np

from . import image
from . import model
from . import server


class Inference(BaseModel):
    esun_uuid: Optional[str] = None
    esun_timestamp: Optional[int] = None
    image: str
    retry: Optional[int] = None


class RunTimeSession(BaseModel):
    name: str
    session: onnxruntime.InferenceSession
    img_size: int

    class Config:
        arbitrary_types_allowed = True


class RunTimePredict(BaseModel):
    name: str
    predict: np.ndarray
    idx_to_word: Dict[str, str]

    class Config:
        arbitrary_types_allowed = True

    def fetch_max_idx(self):
        return np.argmax(self.predict)

    def fetch_max_prob(self):
        return self.predict[self.fetch_max_idx()]

    def fetch_predict_word(self):
        return self.idx_to_word.get(str(self.fetch_max_idx()))

    def log_prediction(self):
        logger.info(f"{self.name}\tpred:{self.fetch_predict_word()} - {self.fetch_max_prob():.4f}")
