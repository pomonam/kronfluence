import functools
import logging
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist

from kronfluence.utils.state import State

TQDM_BAR_FORMAT = (
    "{desc} [{n_fmt}/{total_fmt}] {percentage:3.0f}%|{bar}{postfix} "
    "[time left: {remaining}, time spent: {elapsed}]"
)


class MultiProcessAdapter(logging.LoggerAdapter):
    """An adapter to assist with logging in multiprocess.

    The code is copied from https://github.com/huggingface/accelerate/blob/main/src/accelerate/logging.py with
    minor modifications.
    """

    def log(self, level, msg, *args, **kwargs):
        """Delegates logger call after checking if we should log."""
        if self.isEnabledFor(level) and not self.extra["disable_log"]:
            msg, kwargs = self.process(msg, kwargs)
            self.logger.log(level, msg, *args, **kwargs)

    @functools.lru_cache(None)  # pylint: disable=method-cache-max-size-none
    def warning_once(self, *args, **kwargs):
        """This method is identical to `logger.warning()`, but will emit the warning with the same
        message only once."""
        self.warning(*args, **kwargs)


def get_logger(name: str, disable_log: bool = False, log_level: int = None):
    """Returns the logger with an option to disable."""
    logger = logging.getLogger(name)
    if log_level is not None:
        logger.setLevel(log_level)
        logger.root.setLevel(log_level)
    return MultiProcessAdapter(logger, {"disable_log": disable_log})


def _get_monotonic_time() -> float:
    """Gets the time after CUDA synchronization."""
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()
    return time.monotonic()


class Profiler:
    """Profiling object to measure the time taken to run a certain operation.

    The code is modified from:
    - https://github.com/Lightning-AI/lightning/tree/master/src/pytorch_lightning/profilers
    - https://github.com/mlcommons/algorithmic-efficiency/blob/main/algorithmic_efficiency/profiler.py
    """

    def __init__(self, local_rank: Optional[int] = None) -> None:
        self._local_rank = local_rank
        self.current_actions: Dict[str, float] = {}
        self.recorded_durations = defaultdict(list)
        self.start_time = _get_monotonic_time()

    def set_local_rank(self, local_rank: int) -> None:
        """Sets the current local rank."""
        self._local_rank = local_rank

    @property
    def local_rank(self) -> int:
        """Returns the current local rank."""
        return 0 if self._local_rank is None else self._local_rank

    def start(self, action_name: str) -> None:
        """Start recording the initial time for an action."""
        if self.local_rank != 0:
            pass
        if action_name in self.current_actions:
            raise ValueError(
                f"Attempted to start {action_name} which has already started."
            )
        self.current_actions[action_name] = _get_monotonic_time()

    def stop(self, action_name: str) -> None:
        """Stops recording the initial time for an action."""
        if self.local_rank != 0:
            pass
        end_time = _get_monotonic_time()
        if action_name not in self.current_actions:
            raise ValueError(
                f"Attempting to stop recording an action "
                f"({action_name}) which was never started."
            )
        start_time = self.current_actions.pop(action_name)
        duration = end_time - start_time
        self.recorded_durations[action_name].append(duration)

    @contextmanager
    def profile(self, action_name: str) -> Generator:
        """A context manager for Profiler."""
        try:
            self.start(action_name)
            yield action_name
        finally:
            self.stop(action_name)

    def _make_report(
        self,
    ) -> Tuple[List[Tuple[str, float, float, int, float, float]], int, float]:
        total_duration = _get_monotonic_time() - self.start_time
        report = [
            (
                str(a),
                float(np.mean(d)),
                float(np.std(d)),
                len(d),
                float(np.sum(d)),
                100.0 * float(np.sum(d)) / total_duration,
            )
            for a, d in self.recorded_durations.items()
        ]
        report.sort(key=lambda x: x[5], reverse=True)
        total_calls = sum(x[3] for x in report)
        return report, total_calls, total_duration

    def summary(self) -> str:
        """Returns a formatted summary for the Profiler."""
        sep = os.linesep
        output_string = "Profiler Report:"

        if len(self.recorded_durations) > 0:
            max_key = max(len(k) for k in self.recorded_durations.keys())

            def log_row(action, mean, std, num_calls, total, per):
                row = f"{sep}|  {action:<{max_key}s}\t|  "
                row += f"{mean:<15}\t|  {std:<15}\t|"
                row += f"  {num_calls:<15}\t|  {total:<15}\t|  {per:<15}\t|"
                return row

            header_string = log_row(
                "Action",
                "Mean Duration (s)",
                "Std Duration (s)",
                "Num Calls",
                "Total Time (s)",
                "Percentage %",
            )
            output_string_len = len(header_string.expandtabs())
            sep_lines = f'{sep}{"-" * output_string_len}'
            output_string += sep_lines + header_string + sep_lines
            report, total_calls, total_duration = self._make_report()
            output_string += log_row(
                "Total",
                "-----",
                "-----",
                f"{total_calls:}",
                f"{total_duration:.5}",
                "100 %",
            )
            output_string += sep_lines
            for (
                action,
                mean_duration,
                std_duration,
                num_calls,
                total_duration,
                duration_per,
            ) in report:
                output_string += log_row(
                    action,
                    f"{mean_duration:.5}",
                    f"{std_duration:.5}",
                    f"{num_calls}",
                    f"{total_duration:.5}",
                    f"{duration_per:.5}",
                )
            output_string += sep_lines
        output_string += sep
        return output_string


class PassThroughProfiler(Profiler):
    """A dummy Profiler objective."""

    def start(self, action_name: str) -> None:
        pass

    def stop(self, action_name: str) -> None:
        pass

    def summary(self) -> str:
        return ""


def sync_ddp_time(_time: float, device: torch.device) -> float:
    """Synchronizes the time."""
    time_tensor = torch.tensor(_time, dtype=torch.float64, device=device)
    dist.all_reduce(time_tensor, op=dist.ReduceOp.MAX)
    return time_tensor.item()


def get_time(state: State) -> float:
    """Gets the current time after synchronizing with other devices."""
    if not state.use_distributed:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.time()
    torch.cuda.synchronize()
    t = time.time()
    return sync_ddp_time(t, state.device)
