from .agent import Agent
from typing import Callable, Dict, Any


class OracleAgent(Agent):
    """The OracleAgent gets its act-methode from the task"""

    def __init__(self, act_f: Callable[[Dict, Dict], Dict]) -> None:
        """Initialize an OracleAgent

        Args:
            act_f (Callable[[Dict, Dict], Dict]): An act-callable that produces actions depending on the task, gets defined inside the Task
        """
        super().__init__()
        self.act_f = act_f

    def act(self, obs, info) -> Any:
        return self.act_f(obs, info)
