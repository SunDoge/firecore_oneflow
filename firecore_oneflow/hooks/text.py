from .base import BaseHook


class TextLoggerHook(BaseHook):
    def __init__(
        self,
        interval: int = 100,
    ) -> None:
        super().__init__()
        self._interval = interval


    