from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseModel(ABC):
    """
    Abstract base class for all models.
    """

    @abstractmethod
    def __init__(self, cfg: Any):
        """
        Initializes the model with a configuration object.
        """
        pass

    @abstractmethod
    def run_step(self, env_state: List[str], explore: bool = True) -> Any:
        """
        Executes one step of the model's logic.
        """
        pass

    @abstractmethod
    def apply_learning(self, *args, **kwargs) -> None:
        """
        Applies the learning mechanism of the model.
        """
        pass
