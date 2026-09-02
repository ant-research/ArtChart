from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from os import PathLike

class BaseAPIHandler(ABC):

    @abstractmethod
    def submit_messages(self, messages: List, model_name: str) -> Dict:
        pass

    @staticmethod
    @abstractmethod
    def prepare_messages(image_links: Optional[List[Any]] = None, text_prompt: str = "", force_same_size: bool = False) -> List[Dict]:
        pass

    def _log(self, msg, style="info", end="\n", flush=True):
        """
        Print a styled log message to the console, including current time.
        style: info, success, warning, error
        """
        import datetime
        color = {
            "info": "\033[36m",
            "success": "\033[32m",
            "warning": "\033[33m",
            "error": "\033[31m"
        }.get(style, "\033[0m")
        reset = "\033[0m"
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{color}[{now}] {msg}{reset}", end=end, flush=flush)