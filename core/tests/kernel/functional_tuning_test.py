import logging
import torch
import shark_turbine.kernel as tk
import shark_turbine.kernel.lang as tkl
import shark_turbine.kernel.functional as tkf
import pytest
import subprocess
import time
import os

BLOCK_M = [32, 64, 128, 256]
BLOCK_N = [32, 64, 128, 256]
BLOCK_K = [32, 64, 128]
RATIO_M = [1, 2, 4, 8]
RATIO_N = [1, 2, 4, 8]
RESOURCE_MMA = range(1, 20)
RESOURCE_SHARED = range(1, 20)
RESOURCE_GLOBAL = range(1, 20)
DELAY_MMA = range(1, 20)
DELAY_SHARED = range(1, 20)
DELAY_GLOBAL = range(1, 20)
MATRIX_M = 2048
MATRIX_N = 10240
MATRIX_K = 1280
BUILD_DIR = "/data/home/perf/harsh/iree-build/tools/"


def run_command(command, timeout_limit):
    """
    Constructs the command and executes the function in a separate subprocess, capturing output and error, with a timeout limit using time.sleep.

    Args:
      func_name: The name of the function to call.
      module_name: The name of the module containing the function.
      args: A list of arguments to pass to the function.
      timeout_limit: The maximum execution time in seconds.

    Returns:
      A tuple containing the captured output (decoded string) and any error (decoded string).
    """
    start_time = time.time()
    try:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        while process.poll() is None:  # Check if process is still running
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout_limit:
                process.terminate()  # Try to terminate if timed out
                raise subprocess.TimeoutExpired("Timeout reached")
            time.sleep(0.1)  # Briefly yield to avoid busy waiting
        # Check returncode for successful execution
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command)
        output, error = process.communicate()
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
        output, error = b"", str(e)  # Set output/error for timeout or other errors
    return output.decode().strip(), error.decode() if error else None


# Compile mma.mlir -> mma.vmfb
def compile_to_vmfb():
    cmd = [
        os.path.join(BUILD_DIR, "iree-compile"),
        "SHARK-Turbine/mma.mlir",
        "-o",
        "mma.vmfb",
    ]
    output, error = run_command(cmd, 5.0)


# Run & compare answer
def run_and_validate_result():
    cmd = [
        os.path.join(BUILD_DIR, "iree-run-module"),
    ]
    output, error = run_command(cmd, 5.0)


# Benchmark if correct
def benchmark():
    cmd = [
        os.path.join(BUILD_DIR, "iree-benchmark-module"),
    ]
    output, error = run_command(cmd, 5.0)


# Write result to file
def log_configuration_and_result(x, metric):
    with open("summary.txt", "a") as f:
        str = f"{metric}"
        for val in x:
            str += f",{val}"
        str += "\n"
        f.write(str)


@pytest.mark.parametrize("block_m", BLOCK_M)
@pytest.mark.parametrize("block_n", BLOCK_N)
@pytest.mark.parametrize("block_k", BLOCK_K)
@pytest.mark.parametrize("ratio_m", RATIO_M)
@pytest.mark.parametrize("ratio_n", RATIO_N)
@pytest.mark.parametrize("resource_mma", RESOURCE_MMA)
@pytest.mark.parametrize("resource_shared", RESOURCE_SHARED)
@pytest.mark.parametrize("resource_global", RESOURCE_GLOBAL)
@pytest.mark.parametrize("delay_mma", DELAY_MMA)
@pytest.mark.parametrize("delay_shared", DELAY_SHARED)
@pytest.mark.parametrize("delay_global", DELAY_GLOBAL)
def testGemm(
    block_m,
    block_n,
    block_k,
    ratio_m,
    ratio_n,
    resource_mma,
    resource_shared,
    resource_global,
    delay_mma,
    delay_shared,
    delay_global,
    MATRIX_M,
    MATRIX_N,
    MATRIX_K,
):

    # Wave tile sizes (determined by constraints below)
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K

    # Workgroup tile sizes
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    # Other hyperparameters
    LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
    STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD

    # Expose user-constraints
    constraints = [tkf.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkf.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkf.TilingConstraint(K, BLOCK_K)]
    constraints += [tkf.WaveConstraint(M, BLOCK_M / ratio_m, 0, 64)]
    constraints += [tkf.WaveConstraint(N, BLOCK_N / ratio_n, 1, 64)]
    constraints += [
        tkf.SchedulingConstraint(
            resources={
                "GLOBAL": resource_global,
                "SHARED": resource_shared,
                "MMA": resource_mma,
            },
            delay={"GLOBAL": delay_global, "SHARED": delay_shared, "MMA": delay_mma},
        )
    ]
    constraints += [
        tkf.HardwareConstraint(threads_per_wave=64, mma_type="MFMA_F32_16x16x16_F16")
    ]

    # Wave-level micro-kernel.
    # Since warps are not directly addressable, there is no
    # explicit notion of a warp id (like a workgroup or thread id).
    # Here we use a functional style of expressing the loop since
    # we do not know the loop bounds.
    @tkf.wave(constraints)
    def gemm(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f32],
    ):
        # This microkernel encodes the fact that if the reduction
        # dimension were tiled, then we would need to materialize a loop.
        # c_reg: tkf.Register[WAVE_M, WAVE_N, tkl.f32]
        c_reg = tkf.construct_register_from_metadata((M, N), tkl.f32, 0.0)

        # Do we maybe rather need the info that this is a reduction dimension?
        # This could be called tkf.dim(K) or tkf.reduction(K) ?
        @tkf.tiled_loop(K, init_args=[c_reg])
        def repeat(c_reg) -> tkl.Register[M, N, tkl.f32]:
            # a_reg: tkf.Register[M, K, tkl.f16]
            # b_reg: tkf.Register[N, K, tkl.f16]
            a_reg = tkf.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            b_reg = tkf.read(b, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            c_reg = tkf.mma(a_reg, b_reg, c_reg)
            return c_reg

        # Call removed as the init arg is now explicit above.
        # result = repeat(c_reg)
        tkf.write(repeat, c, elements_per_thread=STORE_ELEMS_PER_THREAD)
        # We also discussed using `repeat` directly in tkf.write:
        # tkf.write(repeat, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    hyperparams = {
        ADDRESS_SPACE: tkl.AddressSpace.SHARED_MEMORY.value,
        LOAD_ELEMS_PER_THREAD: 4,
        STORE_ELEMS_PER_THREAD: 1,
        BLOCK_M: block_m,
        BLOCK_N: block_n,
        BLOCK_K: block_k,
        M: MATRIX_M,
        N: MATRIX_N,
        K: MATRIX_K,
    }
    with tk.gen.TestLaunchContext(hyperparams):
        a = torch.randn(MATRIX_M, MATRIX_K, dtype=torch.float16)
        b = torch.randn(MATRIX_N, MATRIX_K, dtype=torch.float16)
        c = torch.zeros(MATRIX_M, MATRIX_N, dtype=torch.float32)
        gemm(a, b, c)

    # Compile mma.mlir -> mma.vmfb
    success = compile_to_vmfb()

    # Run & compare answer
    if success:
        success = run_and_validate_result()

    # Benchmark if correct
    metric = 0
    if success:
        success, metric = benchmark()

    # Write result to file
    x = [
        block_m,
        block_n,
        block_k,
        ratio_m,
        ratio_n,
        resource_mma,
        resource_shared,
        resource_global,
        delay_mma,
        delay_shared,
        delay_global,
        MATRIX_M,
        MATRIX_N,
        MATRIX_K,
    ]
    log_configuration_and_result(x, metric)
