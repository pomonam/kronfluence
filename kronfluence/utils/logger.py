import logging
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, Generator, List, Tuple

import torch
import torch.distributed as dist
import torch.profiler as t_prof

from kronfluence.utils.state import State

TQDM_BAR_FORMAT = (
    "{desc} [{n_fmt}/{total_fmt}] {percentage:3.0f}%|{bar}{postfix} " "[time left: {remaining}, time spent: {elapsed}]"
)

_TABLE_ROW = Tuple[str, float, int, float, float]
_TABLE_DATA = List[_TABLE_ROW]


class MultiProcessAdapter(logging.LoggerAdapter):
    """An adapter to assist with logging in multiprocess.

    The code is copied from https://github.com/huggingface/accelerate/blob/main/src/accelerate/logging.py with
    minor modifications.
    """

    def log(self, level: int, msg: str, *args, **kwargs) -> None:
        """Delegates logger call after checking if it should log."""
        if self.isEnabledFor(level) and not self.extra["disable_log"]:
            msg, kwargs = self.process(msg, kwargs)
            self.logger.log(level, msg, *args, **kwargs)


def get_logger(name: str, disable_log: bool = False, log_level: int = None) -> MultiProcessAdapter:
    """Returns the logger with an option to disable logging."""
    logger = logging.getLogger(name)
    if log_level is not None:
        logger.setLevel(log_level)
        logger.root.setLevel(log_level)
    return MultiProcessAdapter(logger, {"disable_log": disable_log})


class Profiler:
    """Profiling object to measure the time taken to run a certain operation. The profiler is helpful
    for checking any bottlenecks in the code.

    The code is modified from:
    - https://github.com/Lightning-AI/lightning/tree/master/src/pytorch_lightning/profilers.
    - https://github.com/mlcommons/algorithmic-efficiency/blob/main/algorithmic_efficiency/profiler.py.
    """

    def __init__(self, state: State) -> None:
        """Initializes an instance of the Profiler class.

        Args:
            state (State):
                The current process's information (e.g., device being used).
        """
        self.state = state
        self.current_actions: Dict[str, float] = {}
        self.recorded_durations = defaultdict(list)

    def start(self, action_name: str) -> None:
        """Defines how to start recording an action."""
        if not self.state.is_main_process:
            return
        if action_name in self.current_actions:
            raise ValueError(f"Attempted to start {action_name} which has already started.")
        self.current_actions[action_name] = _get_monotonic_time()

    def stop(self, action_name: str) -> None:
        """Defines how to record the duration once an action is complete."""
        if not self.state.is_main_process:
            return
        end_time = _get_monotonic_time()
        if action_name not in self.current_actions:
            raise ValueError(f"Attempting to stop recording an action " f"({action_name}) which was never started.")
        start_time = self.current_actions.pop(action_name)
        duration = end_time - start_time
        self.recorded_durations[action_name].append(duration)

    @contextmanager
    def profile(self, action_name: str) -> Generator:
        """Yields a context manager to encapsulate the scope of a profiled action."""
        try:
            self.start(action_name)
            yield action_name
        finally:
            self.stop(action_name)

    @torch.no_grad()
    def _make_report(self) -> Tuple[_TABLE_DATA, float, float]:
        total_duration = 0.0
        for a, d in self.recorded_durations.items():
            d_tensor = torch.tensor(d, dtype=torch.float64, requires_grad=False)
            total_duration += torch.sum(d_tensor).item()

        report = []
        for a, d in self.recorded_durations.items():
            d_tensor = torch.tensor(d, dtype=torch.float64, requires_grad=False)
            len_d = len(d)
            sum_d = torch.sum(d_tensor).item()
            percentage_d = 100.0 * sum_d / total_duration
            report.append((a, sum_d / len_d, len_d, sum_d, percentage_d))

        report.sort(key=lambda x: x[4], reverse=True)
        total_calls = sum(x[2] for x in report)
        return report, total_calls, total_duration

    def summary(self) -> str:
        """Returns a formatted summary for the Profiler."""
        sep = os.linesep
        output_string = "Profiler Report:"

        if len(self.recorded_durations) > 0:
            max_key = max(len(k) for k in self.recorded_durations.keys())

            def log_row(action: str, mean: str, num_calls: str, total: str, per: str) -> str:
                row = f"{sep}|  {action:<{max_key}s}\t|  {mean:<15}\t|"
                row += f"  {num_calls:<15}\t|  {total:<15}\t|  {per:<15}\t|"
                return row

            header_string = log_row("Action", "Mean duration (s)", "Num calls", "Total time (s)", "Percentage %")
            output_string_len = len(header_string.expandtabs())
            sep_lines = f"{sep}{'-' * output_string_len}"
            output_string += sep_lines + header_string + sep_lines
            report_extended, total_calls, total_duration = self._make_report()
            output_string += log_row("Total", "-", f"{total_calls:}", f"{total_duration:.5}", "100 %")
            output_string += sep_lines
            for action, mean_duration, num_calls, total_duration, duration_per in report_extended:
                output_string += log_row(
                    action,
                    f"{mean_duration:.5}",
                    f"{num_calls}",
                    f"{total_duration:.5}",
                    f"{duration_per:.5}",
                )
            output_string += sep_lines
        output_string += sep
        return output_string


class PassThroughProfiler(Profiler):
    """A pass through Profiler objective that does not record timing."""

    def start(self, action_name: str) -> None:
        """Defines how to start recording an action."""
        return

    def stop(self, action_name: str) -> None:
        """Defines how to record the duration once an action is complete."""
        return

    def summary(self) -> str:
        """Returns a formatted summary for the Profiler."""
        return ""


class TorchProfiler(Profiler):
    """A PyTorch Profiler objective that provides detailed profiling information:
    https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html.

    This is useful for low-level profiling in PyTorch, and is not used by default.
    """

    def __init__(self, state: State) -> None:
        """Initializes an instance of the PyTorch Profiler class."""
        super().__init__(state=state)
        self.actions: list = []
        self.trace_outputs: list = []
        self._set_up_torch_profiler()

    def start(self, action_name: str) -> None:
        """Defines how to start recording an action."""
        if action_name in self.current_actions:
            raise ValueError(f"Attempted to start {action_name} which has already started.")
        # Set dummy value, since only used to track duplicate actions.
        self.current_actions[action_name] = 0.0
        self.actions.append(action_name)
        self._torch_prof.start()

    def stop(self, action_name: str) -> None:
        """Defines how to stop recording an action."""
        if action_name not in self.current_actions:
            raise ValueError(f"Attempting to stop recording an action " f"({action_name}) which was never started.")
        _ = self.current_actions.pop(action_name)
        self._torch_prof.stop()

    def _set_up_torch_profiler(self) -> None:
        """Creates the PyTorch profiler object with the necessary arguments."""
        self._torch_prof = t_prof.profile(
            activities=[t_prof.ProfilerActivity.CPU, t_prof.ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
            with_flops=False,
            on_trace_ready=self._trace_handler,
        )

    def _trace_handler(self, p) -> None:
        """Adds the PyTorch Profiler trace output to a list once it is ready."""
        # Set metric to sort based on device.
        is_cpu = self.state.device == torch.device("cpu")
        sort_by_metric = "self_cpu_time_total" if is_cpu else "self_cuda_time_total"

        # Obtain formatted output from profiler.
        output = p.key_averages().table(sort_by=sort_by_metric, row_limit=10)
        self.trace_outputs.append(output)

        # Obtain total time taken for the action.
        if is_cpu:
            total_time = p.key_averages().self_cpu_time_total
        else:
            total_time = sum(event.self_cuda_time_total for event in p.key_averages())
        total_time = total_time * 10 ** (-6)  # Convert from micro sec to sec
        self.recorded_durations[self.actions[-1]].append(total_time)

    def _reset_output(self) -> None:
        """Resets actions and outputs list."""
        self.actions = []
        self.trace_outputs = []

    def _high_level_summary(self) -> str:
        """Returns a formatted high level summary for the PyTorch Profiler."""
        sep = os.linesep
        output_string = "Overall PyTorch Profiler Report:"

        if len(self.recorded_durations) > 0:
            max_key = max(len(k) for k in self.recorded_durations.keys())

            def log_row(action: str, mean: str, num_calls: str, total: str, per: str) -> str:
                row = f"{sep}|  {action:<{max_key}s}\t|  {mean:<15}\t|"
                row += f"  {num_calls:<15}\t|  {total:<15}\t|  {per:<15}\t|"
                return row

            header_string = log_row("Action", "Mean duration (s)", "Num calls", "Total time (s)", "Percentage %")
            output_string_len = len(header_string.expandtabs())
            sep_lines = f"{sep}{'-' * output_string_len}"
            output_string += sep_lines + header_string + sep_lines
            report_extended, total_calls, total_duration = self._make_report()
            output_string += log_row("Total", "-", f"{total_calls:}", f"{total_duration:.5}", "100 %")
            output_string += sep_lines
            for action, mean_duration, num_calls, total_duration, duration_per in report_extended:
                output_string += log_row(
                    action,
                    f"{mean_duration:.5}",
                    f"{num_calls}",
                    f"{total_duration:.5}",
                    f"{duration_per:.5}",
                )
            output_string += sep_lines
        output_string += sep
        return output_string

    def summary(self) -> str:
        """Returns a formatted summary for the PyTorch Profiler."""
        assert len(self.actions) == len(self.trace_outputs), (
            "Mismatch in the number of actions and outputs collected: "
            + f"# Actions: {len(self.actions)}, # Ouptuts: {len(self.trace_outputs)}"
        )
        prof_prefix = "Profiler Summary for Action"
        no_summary_str = "*** No Summary returned from PyTorch Profiler ***"
        # Consolidate detailed summary.
        outputs = [no_summary_str if elm == "" else elm for elm in self.trace_outputs]
        summary = "\n".join([f"\n{prof_prefix}: {elm[0]}\n{elm[1]}" for elm in zip(self.actions, outputs)])
        # Append overall action level summary.
        summary = f"{summary}\n\n{self._high_level_summary()}"
        # Reset actions and outputs once summary is invoked.
        self._reset_output()
        return summary


# Timing utilities copied from
# https://github.com/mlcommons/algorithmic-efficiency/blob/main/algorithmic_efficiency/pytorch_utils.py.
def _get_monotonic_time() -> float:
    """Gets the monotonic time after the CUDA synchronization if necessary."""
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()
    return time.monotonic()


@torch.no_grad()
def get_time(state: State) -> float:
    """Gets the current time after synchronizing with other devices."""
    if not state.use_distributed:
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.synchronize()
        return time.time()
    torch.cuda.synchronize()
    current_time = time.time()
    time_tensor = torch.tensor(current_time, dtype=torch.float64, device=state.device, requires_grad=False)
    dist.all_reduce(time_tensor, op=dist.ReduceOp.MAX)
    return time_tensor.item()
