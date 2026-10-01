"""Multi-start insertion search for the Autoclave Queue benchmark."""

from time import perf_counter

from adapter import Solver
from benchkit.rng import Rng, derive_seed


def _cost(instance, order):
    p, r, d, w, fam, setup = (instance.proc, instance.release, instance.due,
                              instance.weight, instance.fam, instance.setup)
    t = prev = None
    tardy = 0
    for b in order:
        t = (r[b] if prev is None else max(t + setup[prev][fam[b]], r[b])) + p[b]
        if t > d[b]:
            tardy += w[b] * (t - d[b])
        prev = fam[b]
    return 4 * t + tardy


def _insert(order, src, dst):
    item = order[src]
    result = order[:src] + order[src + 1:]
    result.insert(dst, item)
    return result


def _descend(instance, order, deadline):
    order = list(order)
    score = _cost(instance, order)
    n = len(order)
    while perf_counter() < deadline:
        best_score, best_move = score, None
        for src in range(n):
            without = order[:src] + order[src + 1:]
            item = order[src]
            for dst in range(n):
                if dst == src:
                    continue
                candidate = without[:dst] + [item] + without[dst:]
                value = _cost(instance, candidate)
                if value < best_score:
                    best_score, best_move = value, candidate
        if best_move is None:
            break
        order, score = best_move, best_score
    return order, score


class InsertionSolver(Solver):
    def solve(self, instance, submit_candidate):
        n = instance.size
        rng = Rng(derive_seed("autoclave-insertion", instance.digest))
        due = instance.due
        weight = instance.weight

        # A few distinct dispatch rules provide useful starting basins.
        seeds = [
            sorted(range(n), key=lambda b: (due[b], b)),
            sorted(range(n), key=lambda b: (due[b] - instance.proc[b], due[b], b)),
            sorted(range(n), key=lambda b: ((due[b] - instance.proc[b]) / weight[b], b)),
            sorted(range(n), key=lambda b: (instance.release[b], due[b], b)),
        ]
        best = seeds[0]
        best_cost = _cost(instance, best)
        submit_candidate({"order": best})

        started = perf_counter()
        deadline = started + 4.72
        for seed in seeds[1:]:
            value = _cost(instance, seed)
            if value < best_cost:
                best, best_cost = seed, value
                submit_candidate({"order": best})

        current = list(best)
        while perf_counter() < deadline:
            candidate, value = _descend(instance, current, deadline)
            if value < best_cost:
                best, best_cost = candidate, value
                submit_candidate({"order": best})
            # Iterated local search: perturb a local optimum and search its basin.
            current = list(candidate)
            count = max(2, n // 12)
            for _ in range(count):
                src = rng.below(n)
                dst = rng.below(n)
                current = _insert(current, src, dst)
            if perf_counter() < deadline:
                value = _cost(instance, current)
                if value < best_cost:
                    best, best_cost = list(current), value
                    submit_candidate({"order": best})

        return {"order": best}
