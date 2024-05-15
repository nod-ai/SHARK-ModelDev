import logging
import torch
import shark_turbine.kernel as tk
import shark_turbine.kernel.lang as tkl
import shark_turbine.kernel.functional as tkf
import pytest
import subprocess
import time
import os
import numpy as np

BLOCK_M = [32, 64, 128, 256]
BLOCK_N = [32, 64, 128, 256]
BLOCK_K = [32, 64, 128]
RATIO_M = [1, 2, 4, 8]
RATIO_N = [1, 2, 4, 8]
RESOURCE_MMA = list(range(1, 20))
RESOURCE_SHARED = list(range(1, 20))
RESOURCE_GLOBAL = list(range(1, 20))
DELAY_MMA = list(range(1, 20))
DELAY_SHARED = list(range(1, 20))
DELAY_GLOBAL = list(range(1, 20))
MATRIX_M = 2048
MATRIX_N = 10240
MATRIX_K = 1280
BUILD_DIR = "/data/home/perf/harsh/iree-build/tools/"
DEVICE = 4
TIMEOUT = 60

BLOCK_M=[128]
BLOCK_N=[128]
BLOCK_K=[32]
RATIO_M=[1]
RATIO_N=[1]
RESOURCE_MMA=[2]
RESOURCE_SHARED=[2]
RESOURCE_GLOBAL = [2]
DELAY_MMA = [2]
DELAY_SHARED = [1]
DELAY_GLOBAL = [5]
MMA_INSTRUCTION = ['MFMA_F32_16x16x16_F16', 'MFMA_F32_32x32x8_F16']
MMA_INSTRUCTION = ['MFMA_F32_16x16x16_F16']


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
    print('\n' + ' '.join(command) + '\n')
    start_time = time.time()
    try:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        while process.poll() is None:  # Check if process is still running
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout_limit:
                process.terminate()  # Try to terminate if timed out
                raise subprocess.TimeoutExpired("Timeout reached", timeout_limit)
        # Check returncode for successful execution
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command)
        output, error = process.communicate()
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
        output, error = b"", str(e)  # Set output/error for timeout or other errors
    return output.strip(), error if error else None


# Compile mma.mlir -> mma.vmfb
def compile_to_vmfb():
    cmd = [
        os.path.join(BUILD_DIR, "iree-compile"),
        "/data/home/perf/harsh/SHARK-Turbine/mma.mlir", \
        "--iree-hal-target-backends=rocm", \
        "--iree-rocm-target-chip=gfx942", \
        "--iree-rocm-bc-dir=/opt/rocm/amdgcn/bitcode", \
        "--iree-hal-benchmark-dispatch-repeat-count=1000", \
        "--iree-hal-dump-executable-intermediates-to=intermediates/gemm", \
        "-o", \
        "mma.vmfb",
    ]
    output, error = run_command(cmd, TIMEOUT)
    return error is None


# Run & compare answer
def run_and_validate_result():
    cmd = [
        os.path.join(BUILD_DIR, "iree-run-module"),
        f"--device=rocm://{DEVICE}", \
        "--device_allocator=caching", \
        "--module=mma.vmfb", \
        "--function=isolated_benchmark", \
        f"--input=@/data/home/perf/harsh/mma_files/a_matrix_{MATRIX_M}x{MATRIX_N}x{MATRIX_K}.npy", \
        f"--input=@/data/home/perf/harsh/mma_files/b_matrix_{MATRIX_M}x{MATRIX_N}x{MATRIX_K}.npy", \
        f"--output=@/data/home/perf/harsh/mma_files/output_{MATRIX_M}x{MATRIX_N}x{MATRIX_K}.npy"
    ]
    output, error = run_command(cmd, TIMEOUT)
    if error is not None:
        return False
    computed = np.load(os.path.join(os.getcwd(), f"/data/home/perf/harsh/mma_files/output_{MATRIX_M}x{MATRIX_N}x{MATRIX_K}.npy"))
    reference = np.load(os.path.join(os.getcwd(), f"/data/home/perf/harsh/mma_files/iree_ref_{MATRIX_M}x{MATRIX_N}x{MATRIX_K}.npy"))
    max_error = np.max(np.abs(computed - reference))
    print("Max error = ", max_error)
    return max_error == 0.0


# Benchmark if correct
def benchmark():
    cmd = [
        os.path.join(BUILD_DIR, "iree-benchmark-module"),
        f"--device=rocm://{DEVICE}", \
        "--device_allocator=caching", \
        "--module=mma.vmfb", \
        "--function=isolated_benchmark", \
        "--batch_size=1000", \
        "--benchmark_repetitions=3", \
        f"--input={MATRIX_M}x{MATRIX_K}xf16", \
        f"--input={MATRIX_N}x{MATRIX_K}xf16"
    ]
    output, error = run_command(cmd, TIMEOUT)
    decoded_output = output.decode('utf-8')
    metric = 0
    if 'Benchmark' in decoded_output:
        metric = [x for x in decoded_output.split('\n')[3].split(' ') if x][1]
    return metric


# Write result to file
def log_configuration_and_result(x, metric):
    with open(f"summary_{MATRIX_M}_{MATRIX_N}_{MATRIX_K}.txt", "a") as f:
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
@pytest.mark.parametrize("mma_instruction", MMA_INSTRUCTION)
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
    mma_instruction
):

    # Wave tile sizes (determined by constraints below)
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K

    # Workgroup tile sizes
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    # MMA tile sizes
    MMA_M = tkl.sym.MMA_M
    MMA_N = tkl.sym.MMA_N
    MMA_K = tkl.sym.MMA_K
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    # Other hyperparameters
    LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
    STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD
    GLOBAL_LOAD_ELEMS_PER_THREAD = tkl.sym.GLOBAL_LOAD_ELEMS_PER_THREAD

    # Expose user-constraints
    constraints = [tkf.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkf.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkf.TilingConstraint(K, BLOCK_K)]
    constraints += [tkf.WaveConstraint(M, BLOCK_M / ratio_m, 0, 64)]
    constraints += [tkf.WaveConstraint(N, BLOCK_N / ratio_n, 1, 64)]
    constraints += [
        tkf.SchedulingConstraint(
            # Resource
            {
                "GLOBAL": resource_global,
                "SHARED": resource_shared,
                "MMA": resource_mma,
            },
            # Delays
            {"GLOBAL": delay_global, "SHARED": delay_shared, "MMA": delay_mma},
        )
    ]
    constraints += [
        tkf.HardwareConstraint(threads_per_wave=64, mma_type=mma_instruction)
    ]
    mma_m = mma_n = mma_k = 16
    if mma_instruction == 'MFMA_F32_32x32x8_F16':
        mma_m = mma_n = 32
        mma_k = 8

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
        GLOBAL_LOAD_ELEMS_PER_THREAD: 8,
        BLOCK_M: block_m,
        BLOCK_N: block_n,
        BLOCK_K: block_k,
        MMA_M: mma_m,
        MMA_N: mma_n,
        MMA_K: mma_k,
        M: MATRIX_M,
        N: MATRIX_N,
        K: MATRIX_K,
    }
    try:
        with tk.gen.TestLaunchContext(hyperparams):
            a = torch.randn(MATRIX_M, MATRIX_K, dtype=torch.float16)
            b = torch.randn(MATRIX_N, MATRIX_K, dtype=torch.float16)
            c = torch.zeros(MATRIX_M, MATRIX_N, dtype=torch.float32)
            gemm(a, b, c)
    except:
        return

    # Compile mma.mlir -> mma.vmfb
    success = compile_to_vmfb()

    # Run & compare answer
    if success:
        success = run_and_validate_result()

    # Benchmark if correct
    metric = benchmark()
    print("metric = ", metric)

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
        mma_instruction,
        MATRIX_M,
        MATRIX_N,
        MATRIX_K,
    ]
    log_configuration_and_result(x, metric)
