import subprocess
from collections import namedtuple
import iree.runtime as ireert
import numpy as np


BenchmarkResult = namedtuple(
    "BenchmarkResult", "benchmark_name time cpu_time iterations user_counters"
)


class BenchmarkToolError(Exception):
    """Benchmark exception that preserves the command line and error output."""

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class BenchmarkTimeoutError(Exception):
    """Exception raised if the benchmark is cancelled by the user specified timeout."""

    pass


DTYPE_TO_ABI_TYPE = {
    np.dtype(np.float32): "f32",
    np.dtype(np.float16): "f16",
    np.dtype(np.int32): "i32",
    np.dtype(np.int64): "i64",
    np.dtype(np.float64): "f64",
    np.dtype(np.int16): "i16",
    np.dtype(np.int8): "i8",
    np.dtype(np.bool_): "i1",
}


def benchmark_module(
    module,
    entry_function=None,
    vmfbs=[],
    inputs=[],
    tracy_profile=False,
    timeout=None,
    **kwargs,
):
    funcs = [a for a in module.function_names if a != "__init"]
    if entry_function is None:
        if len(funcs) > 1:
            raise ValueError(f"No function specified with multiple options {funcs}")
        entry_function = funcs[0]
    if entry_function not in funcs:
        raise ValueError(
            f"Attempted to benchmark unknown function {entry_function} of options {funcs}"
        )

    args = []
    if tracy_profile:
        args.append("TRACY_NO_EXIT=1")
        # TODO: run iree-tracy-capture subprocess
    args.append(ireert.benchmark_exe())
    args.append(f"--function={entry_function}")

    for inp in inputs:
        if isinstance(inp, str):
            args.append(f"--input={inp}")
            continue
        shape = "x".join([str(d) for d in inp.shape])
        abitype = DTYPE_TO_ABI_TYPE[inp.dtype]
        values = inp.flatten()
        if np.all(values[0] == values):
            values = str(values[0])
        else:
            values = ",".join([str(v) for v in values])
        input_arg = f"--input={shape}x{abitype}={values}"
        if len(input_arg) > 256:
            print(
                f"Randomizing {input_arg.split('=')[0]} because it is too long for subprocess.run"
            )
            input_arg = f"--input={shape}x{abitype}"
        args.append(input_arg)
        print(args)

    for k in kwargs:
        v = kwargs[k]
        args.append(f"--{k}={v}")

    for vmfb in vmfbs:
        args.append(f"--module={vmfb}")

    try:
        benchmark_process = subprocess.run(
            args=args,
            # input=flatbuffer,
            timeout=timeout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.TimeoutExpired:
        raise BenchmarkTimeoutError(f"Benchmark timed out after {timeout} seconds")
    out = benchmark_process.stdout
    err = benchmark_process.stderr

    err = err.decode()
    if "INVALID_ARGUMENT;" in err:
        raise ValueError("Invalid inputs specified for benchmarking")

    # In the event benchmarking runs but encounteres an internal error,
    # return the internal error instead of benchmark results.
    if "INTERNAL; CUDA driver error" in str(out):
        raise BenchmarkToolError(str(out))

    # Grab individual results by line (skip header lines)
    bench_lines = out.decode().split("\n")[3:]
    benchmark_results = []
    for line in bench_lines:
        split = line.split()
        if len(split) == 0:
            continue
        benchmark_name = split[0]
        time = " ".join(split[1:3])
        cpu_time = " ".join(split[3:5])
        iterations = split[5]
        user_counters = None
        if len(split) > 5:
            user_counters = split[6]
        benchmark_results.append(
            BenchmarkResult(
                benchmark_name=benchmark_name,
                time=time,
                cpu_time=cpu_time,
                iterations=iterations,
                user_counters=user_counters,
            )
        )

    return benchmark_results
