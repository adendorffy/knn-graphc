import concurrent.futures
import multiprocessing as mp
from abc import ABC, abstractmethod
from typing import Any

from tqdm import tqdm


class ConcurrentNArySearcher(ABC):
    """
    Class to abstract away the N-ary search logic and multiprocessing execution.
    We search for some target output by searching the input space over two bounds (min and max).
    Override self.compute_state(input) and self.output_from_state(state) to implement the function to be searched.
    This composed function, output = self.output_from_state(self.compute_state(input)), must be monotonic.
    """

    def __init__(
        self,
        output_target: float,
        output_tolerance: float,
        input_init_low: float,
        input_init_high: float,
        max_iterations: int,
        worker_count: int,
    ):
        assert worker_count >= 1
        assert input_init_low < input_init_high

        self.output_target = output_target
        self.output_tolerance = output_tolerance
        self.input_init_low = input_init_low
        self.input_init_hight = input_init_high
        self.worker_count = worker_count
        self.max_iterations = max_iterations

        self.best_input = 0.0
        self.best_output = 0.0
        self.best_diff = float("inf")

    def setup_context(self, context: Any = None):
        """Override to inject heavy data into global scope before forking."""
        pass

    def teardown_context(self, context: Any = None):
        """Override to clean up global scope after search finishes."""
        pass

    # this must always be overridden and will compute the 'result'
    @abstractmethod
    def compute_state(self, input: float) -> Any:
        """
        Runs in the worker process. Does the heavy lifting for a given input.

        The `state` you return is basically your 'result artifact'. It has two jobs:
        1. Source of Truth: `output_from_state` pulls the scalar target value from it.
        2. Keepable Data: Whatever you return here gets pickled back to the main
           process. Keep it light—if you need to cache big models or data buffers,
           do it via `setup_context` so they stay in the worker's global scope.

        Pro-tip: If you find the optimal input, this `state` is returned from
        `search`. Make sure it’s serializable and contains everything you need to
        debug or validate that point later.
        """
        pass

    def output_from_state(self, state: Any) -> float:
        """
        Extract the scaler `output` associated with the computed `state`.
        """
        return state

    def on_point_evaluated(
        self, iteration: int, point: float, state: Any, value: float, diff: float
    ):
        """Hook for custom logging."""
        pass

    def search(
        self, context: Any = None, show_progress: bool = True
    ) -> tuple[float, float, float, Any]:
        """
        Start the search loop.

        Returns (best_input, best_output, best_diff, best_state).
        """

        self.setup_context(context)
        ctx = mp.get_context("fork")
        best_state: Any = None

        try:
            current_low = self.input_init_low
            current_high = self.input_init_hight
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=self.worker_count, mp_context=ctx
            ) as executor:
                for i in tqdm(
                    range(self.max_iterations),
                    desc="N-ary concurrent search",
                    disable=not show_progress,
                ):
                    step = (current_high - current_low) / (self.worker_count + 1)
                    inputs = [
                        current_low + step * (w + 1) for w in range(self.worker_count)
                    ]

                    future_to_input = {
                        executor.submit(self.compute_state, inp): inp for inp in inputs
                    }

                    results = []
                    for future in concurrent.futures.as_completed(future_to_input):
                        inp = future_to_input[future]
                        try:
                            state = future.result()
                            results.append((inp, state))
                        except Exception as exc:
                            print(f"Exception evaluating point {inp:.6f}: {exc}")

                    assert results

                    results.sort(key=lambda x: x[0])

                    # evaluate results against N-ary boundaries
                    target_reached = False
                    for inp, state in results:
                        outp = self.output_from_state(state)
                        diff = outp - self.output_target

                        self.on_point_evaluated(i, inp, state, outp, diff)

                        abs_diff = abs(diff)

                        if abs_diff < self.best_diff:
                            self.best_diff = abs_diff
                            self.best_input = inp
                            self.best_output = outp
                            best_state = state

                        if abs_diff <= self.output_tolerance:
                            target_reached = True

                    if target_reached:
                        if show_progress:
                            print(
                                f"Target reached! Found best input: {self.best_input:.6f}"
                            )
                        break

                    # narrow the search scope
                    lower_candidates = [
                        inp
                        for inp, state in results
                        if self.output_from_state(state) < self.output_target
                    ]
                    higher_candidates = [
                        inp
                        for inp, state in results
                        if self.output_from_state(state) >= self.output_target
                    ]

                    new_low = max(lower_candidates) if lower_candidates else current_low
                    new_high = min(higher_candidates) if higher_candidates else current_high

                    if new_low >= new_high:
                        # Local non-monotonicity detected.
                        # Shrink around the best point we've seen instead.
                        current_low = max(
                            self.input_init_low,
                            self.best_input - step,
                        )
                        current_high = min(
                            self.input_init_hight,
                            self.best_input + step,
                        )

                        print(
                            f"Non-monotonicity detected. "
                            f"Searching locally around {self.best_input:.6f}"
                        )
                    else:
                        current_low = new_low
                        current_high = new_high

            return self.best_input, self.best_output, self.best_diff, best_state
        finally:
            self.teardown_context(context)
