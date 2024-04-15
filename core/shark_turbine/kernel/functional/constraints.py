from abc import ABC, abstractmethod
import shark_turbine.kernel.lang as tkl
import torch.fx as fx
import math

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
        self.workgroup_ids = [
            tkl.sym.WG0,
            tkl.sym.WG1,
        ]
        self.thread_ids = [
            tkl.sym.TX,
            tkl.sym.TY,
            tkl.sym.TZ,
        ]
        # This is populated when we encounter a tkf.tiled_loop in the graph
        self.induction_variables = []
"""
A constraint of the form
    tkf.WorkgroupConstraint(M, BLOCK_M, 0)
specifies that we want to distribute dimension M along workgroup dim 0
with a tile size of BLOCK_M resulting in M // BLOCK_M workgroups along that
dimension. This translates to an index constraint for all tensors of the
shape [M, ?] -> index += (workgroup_id_0 * BLOCK_M, 0)
"""
class WorkgroupConstraint(ConstraintsMeta):
    def __init__(self, dim, tile_size, workgroup_dim) -> None:
        super().__init__()
        self.dim = dim
        self.tile_size = tile_size
        self.workgroup_dim = workgroup_dim

    def apply(self):
        wg_dim = None
        match self.workgroup_dim:
            case 0: wg_dim = self.workgroup_ids[0]
            case 1: wg_dim = self.workgroup_ids[1]
            case _: raise ValueError("Invalid workgroup index. Expected 0 or 1")
        return wg_dim * self.tile_size


"""
A constraint of the form
    tkf.TilingConstraint(K, BLOCK_K)
specifies that we want to tile the K dimension with a tile size of BLOCK_K.
In the micro-kernel, there will need to be a tkf.tiled_loop that maps to
the same dimension. This translates to the following index constraint
shape[?, K] -> index += (0, arg0 * BLOCK_K)
where arg0 is the induction variable of the tkf.tiled_loop.
"""
class TilingConstraint(ConstraintsMeta):
    def __init__(self, dim, tile_size) -> None:
        super().__init__()
        self.dim = dim
        self.tile_size = tile_size

    def apply(self, induction_var):
        return induction_var * self.tile_size

"""
A constraint of the form
    tkf.ThreadConstraint(threads_per_block = [tx, ty, tz])
specifies that we want to distribute to threads as per the threads per block specification.
By itself, this imposes no index constraints. It needs to be coupled
with a hardware constraint.
"""
class ThreadConstraint(ConstraintsMeta):
    def __init__(self, threads_per_block) -> None:
        super().__init__()
        self.threads_per_block = threads_per_block


"""
A constraint of the form
    tkf.HardwareConstraint(threads_per_wave = N,
                           mma_type = 'MFMA_F32_16x16x16_F16')
specifies that the hardware supports N threads per wave and that
we want all mma operations in the microkernel to be
mapped to a hardware mma instruction of shape (M, N, K).
This translates to a hardware specific index constraint.
"""
class HardwareConstraint(ConstraintsMeta):
    def __init__(self, threads_per_wave, mma_type, threads_per_block=None) -> None:
        super().__init__()
        self.threads_per_block = threads_per_block
        self.threads_per_wave = threads_per_wave
        self.mma_type = mma_type

    def mma_indices(self, mma_type):
        # TODO: Add support for more instructions
        if mma_type == 'MFMA_F32_16x16x16_F16':
            indices = {
                'A': lambda lane, gpr: (lane % 16, 4 * (lane / 16) + gpr),
                'B': lambda lane, gpr: (lane % 16, 4 * (lane / 16) + gpr),
                'C': lambda lane, gpr: (4 * (lane / 16) + gpr, lane % 16)
            }
        return indices

    def get_vector_shape(self, matrix_type):
        if self.mma_type == 'MFMA_F32_16x16x16_F16':
            return 4
        return None

    def apply(self, matrix_type):
        indices = self.mma_indices(self.mma_type)
        lane = (self.thread_ids[0] \
             + self.thread_ids[1] * self.threads_per_block[0] \
             + self.thread_ids[2] * self.threads_per_block[0] * self.threads_per_block[1]) \
             % self.threads_per_wave
        gpr = 0
        return indices[matrix_type](lane, gpr)