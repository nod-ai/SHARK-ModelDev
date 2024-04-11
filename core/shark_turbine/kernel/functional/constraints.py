from abc import ABC, abstractmethod

"""
Base class for constraints. Every constraint reduces to
the following form:
    Variables: [x0, x1, ...., xN]
    Bounds: [lb0 <= x0 <= ub0, ..., lbN <= xN <= ubN]
    Equality Constraints: [f0(x0, ..., xN) = 0, f1(x0, ..., xN) = 0, ...]
    Inequality Constraints: [g0(x0, ..., xN) <= 0, g1(x0, ..., xN) <= 0, ...]
"""
class ConstraintsMeta(ABC):
    def __init__(self) -> None:
        pass

"""
This class imposes a constraint on the workgroup id 0, 1, 2.
Specifically, given a constraint of the form
    tkf.distribution.workgroup_constraint(
        # Tile M dimension with a tile size of BLOCK_M along
        # workgroup 0, respecting the mapping from tile id -> workgroup id 0.
        # (In this case, tile i goes to id i).
        M : (BLOCK_M, 0, lambda i : i),
        # We can repeat this for other dimensions and with more exotic
        # mapping functions.
        N : (BLOCK_N, 1, lambda i : (i + 1) % N),
    )
The SMT solver generates code to read/write from/to the appropriate tiles
based on this constraint. Also computes the number of workgroups.
"""
class WorkgroupConstraint(ConstraintsMeta):
    def __init__(self) -> None:
        super().__init__()