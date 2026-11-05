"""Certify squared bucket-load balance with exact rational weights."""

from decimal import Decimal
from fractions import Fraction
import hashlib
import math

import pandas as pd

from grouping_schemes import lpt_buckets
from mgp_core import RUN_OUTPUT_CACHE, atomic_write_pickle

TOLERANCE = Fraction(1, 10**20)


def exact_lower_bound(weights, m):
    """The squared-load lower bound with the largest jobs alone in buckets."""
    sorted_weights = sorted(weights, reverse=True)
    remaining = sum(sorted_weights)
    fixed = 0
    while fixed < m - 1 and sorted_weights[fixed] * (m - fixed) > remaining:
        remaining -= sorted_weights[fixed]
        fixed += 1
    return sum(weight**2 for weight in sorted_weights[:fixed]) + remaining**2 / (m - fixed)


def squared_load_objective(weights, buckets):
    return sum(sum(weights[i] for i in bucket)**2 for bucket in buckets)


def search_compustat_partition(weights, buckets, lower):
    from ortools.sat.python import cp_model

    integer_weights = list(map(int, weights))
    fixed = [bucket for bucket in buckets if len(bucket) == 1]
    starting_groups = sorted(
        [bucket for bucket in buckets if len(bucket) > 1],
        key=lambda bucket: sum(integer_weights[i] for i in bucket),
        reverse=True,
    )
    remaining = [i for bucket in starting_groups for i in bucket]
    group_count = len(starting_groups)
    total = sum(integer_weights[i] for i in remaining)
    starting_deviations = [
        group_count * sum(integer_weights[i] for i in bucket) - total
        for bucket in starting_groups
    ]
    cap = max(1, sum(value**2 for value in starting_deviations))
    radius = math.isqrt(cap)
    assert group_count * cap < 2**63 - 1
    assert group_count * sum(integer_weights) < 2**63 - 1

    model = cp_model.CpModel()
    assignments = [
        [model.new_bool_var(f"x_{i}_{j}") for j in range(group_count)]
        for i in remaining
    ]
    for choices in assignments:
        model.add_exactly_one(choices)
    deviations = []
    squares = []
    for j in range(group_count):
        deviation = model.new_int_var(-radius, radius, f"e_{j}")
        load = sum(integer_weights[i] * choices[j] for i, choices in zip(remaining, assignments))
        model.add(deviation == group_count * load - total)
        square = model.new_int_var(0, cap, f"z_{j}")
        model.add_multiplication_equality(square, [deviation, deviation])
        deviations.append(deviation)
        squares.append(square)
    for previous, current in zip(deviations, deviations[1:]):
        model.add(previous >= current)
    model.add(sum(squares) <= cap)
    model.minimize(sum(squares))
    starting_assignment = {i: j for j, bucket in enumerate(starting_groups) for i in bucket}
    for i, choices in zip(remaining, assignments):
        for j, choice in enumerate(choices):
            model.add_hint(choice, int(starting_assignment[i] == j))
    for deviation, square, value in zip(deviations, squares, starting_deviations):
        model.add_hint(deviation, value)
        model.add_hint(square, value**2)

    class StopAtTolerance(cp_model.CpSolverSolutionCallback):
        def __init__(self):
            super().__init__()
            self.buckets = None
            self.objective = None

        def on_solution_callback(self):
            candidate = fixed + [
                [i for i, choices in zip(remaining, assignments) if self.value(choices[j])]
                for j in range(group_count)
            ]
            objective = squared_load_objective(weights, candidate)
            if (objective - lower) / lower <= TOLERANCE:
                self.buckets = candidate
                self.objective = objective
                self.stop_search()

    callback = StopAtTolerance()
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60
    solver.parameters.num_search_workers = 4
    status = solver.status_name(solver.solve(model, callback))
    if callback.buckets is None:
        raise RuntimeError(f"CP-SAT did not certify a Compustat partition ({status})")
    gap = (callback.objective - lower) / lower
    print(f"[grouping] CP-SAT {status}, {solver.wall_time:.1f}s, certified gap {float(gap):.3g}", flush=True)
    return callback.buckets, {
        "certified_relative_gap": str(gap),
        "lower_bound": str(lower),
        "objective": str(callback.objective),
    }


def certified_compustat_allocation(nominal_sizes):
    cache = RUN_OUTPUT_CACHE / "grouping_tolerance_1e20.pkl"
    nominal = [Fraction(Decimal(str(x))) for x in nominal_sizes]
    unit = math.lcm(*(x.denominator for x in nominal))
    weights = [Fraction(int(x * unit) ** 2) for x in nominal]
    signature = hashlib.sha256(str(weights).encode()).hexdigest()
    if cache.exists():
        previous = pd.read_pickle(cache)["compustat"]
        if previous["info"].get("weights_sha256") == signature:
            print(f"[grouping] Using {cache}", flush=True)
            return previous
    starting_buckets = lpt_buckets(nominal_sizes, 10)[0]
    buckets, info = search_compustat_partition(weights, starting_buckets, exact_lower_bound(weights, 10))
    info["weights_sha256"] = signature
    allocation = {"buckets": buckets, "info": info}
    atomic_write_pickle({"compustat": allocation}, cache)
    return allocation
