from .agent import Agent
from typing import Callable, Dict, Any


class OracleAgent(Agent):
    """The OracleAgent gets its act-methode from the task"""

    def __init__(self) -> None:
        """Initialize an OracleAgent

        Args:
            act_f (Callable[[Dict, Dict], Dict]): An act-callable that produces actions depending on the task, gets defined inside the Task
        """
        super().__init__()

    def set_act_f(self, act_f: Callable[[Dict, Dict], Dict]):
        self.act_f = act_f

    def act(self, obs, info) -> Any:
        if self.act_f:
            return self.act_f(obs, info)
        raise RuntimeError(
            "Please set action function first with 'OracleAgent.set_act_f(Callable)'")
