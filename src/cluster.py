from dataclasses import dataclass
from typing import NamedTuple

import igraph as ig
import leidenalg as la

from .nary import ConcurrentNArySearcher


from dataclasses import dataclass

from dataclasses import dataclass

@dataclass
class ClusterConfig:
    cluster_count_target: int | None = None
    cluster_count_tolerance: int = 10

    resolution_specified: float | None = None

    resolution_init_low: float = 0.0
    resolution_init_high: float = 1.0

    search_max_iterations: int = 15
    search_worker_count: int = 4

    leiden_iterations: int = 1
    seed: int | None = 42

    def __post_init__(self):
        if self.resolution_specified is not None:
            self.resolution_init_low = self.resolution_specified - 0.1
            self.resolution_init_high = self.resolution_specified + 0.1

# pickling and passing the graph to all workers is inefficient, so let's make it global context
# we won't change the graph, so the copy-on-write policy will save somne memory here
SHARED_GRAPH = None


# passing the search result with simpler datatypes avoids some issues i had with pickling the CPMVertexPartition
class ClusterSearchResult(NamedTuple):
    cluster_count: int
    membership: list[int]


class ClusterResolutionSearcher(ConcurrentNArySearcher):
    """
    Subclass that applies graph clustering to the single-point evaluation.
    """

    def __init__(self, config: ClusterConfig, show_progress: bool = True):
        super().__init__(
            output_target=config.cluster_count_target,
            output_tolerance=config.cluster_count_tolerance,
            input_init_low=config.resolution_init_low,
            input_init_high=config.resolution_init_high,
            max_iterations=config.search_max_iterations,
            worker_count=config.search_worker_count,
        )

        # store lightweight stuff here
        self.config = config
        self.show_progress = show_progress

    def setup_context(self, context: ig.Graph):
        # store heavy stuff here as globals
        global SHARED_GRAPH
        SHARED_GRAPH = context

    def teardown_context(self, context: ig.Graph | None = None):
        global SHARED_GRAPH
        SHARED_GRAPH = None

    def compute_state(self, input: float) -> ClusterSearchResult:
        global SHARED_GRAPH
        partition = la.find_partition(
            graph=SHARED_GRAPH,
            partition_type=la.CPMVertexPartition,
            initial_membership=None,
            weights="weight",
            resolution_parameter=input,
            n_iterations=self.config.leiden_iterations,
            seed=self.config.seed,
        )
        return ClusterSearchResult(
            cluster_count=len(partition), membership=list(partition.membership)
        )

    def output_from_state(self, state: ClusterSearchResult) -> float:
        return state.cluster_count

    def on_point_evaluated(
        self,
        iteration: int,
        point: float,
        state: ClusterSearchResult,
        value: float,
        diff: float,
    ):
        if self.show_progress:
            print(
                f"[Iter {iteration + 1:02d}] res={point:.6f} | clusters={value:,} | diff={diff:+.0f}"
            )


def partition_with_target_clusters(
    graph: ig.Graph,
    config: ClusterConfig,
    show_progress: bool = True,
) -> tuple[la.CPMVertexPartition, float]:

    if show_progress:
        print(
            f"Optimizing resolution from {config.resolution_init_low:.6f} to {config.resolution_init_high:.6f} "
            f"to find ~{config.cluster_count_target} clusters (±{config.cluster_count_tolerance}) "
            f"using {config.search_worker_count} workers..."
        )

    if config.resolution_specified is not None:
        if show_progress:
            print(
                f"Resolution specified: {config.resolution_specified:.6f}. Skipping search."
            )
        best_res = config.resolution_specified
        best_state = la.find_partition(
            graph=graph,
            partition_type=la.CPMVertexPartition,
            initial_membership=None,
            weights="weight",
            resolution_parameter=best_res,
            n_iterations=config.leiden_iterations,
            seed=config.seed,
        )
        best_output = len(best_state)
        if config.cluster_count_target is not None:
            best_diff = abs(best_output - config.cluster_count_target)
        else:
            best_diff = None
    else:
        searcher = ClusterResolutionSearcher(config, show_progress=show_progress)

        best_res, best_output, best_diff, best_state = searcher.search(
            context=graph, show_progress=show_progress
        )

        if (
            show_progress
            and best_diff is not None
            and best_diff > config.cluster_count_tolerance
        ):
            print(
                f"Max iterations reached. Best diff was {best_diff} at res={best_res:.6f}"
            )

    assert best_state is not None

    final_partition = la.CPMVertexPartition(
        graph=graph,
        weights="weight",
        resolution_parameter=best_res,
        initial_membership=best_state.membership,
    )

    if show_progress:
        print(f"\nBest partition: {best_output:,} clusters at res={best_res:.6f}")

    return final_partition, best_res
