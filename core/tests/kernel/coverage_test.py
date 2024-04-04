import torch
import shark_turbine.kernel as tk
import shark_turbine.kernel.lang as tkl
import pytest


FLOAT_DTYPES = [tkl.f16, tkl.f32, tkl.f64]
INT_DTYPES = [
    tkl.bool,
    tkl.i8,
    tkl.i16,
    tkl.i32,
    tkl.i64,
]


def rms_norm_krnl(dtype, input, weight, output):
    M = tkl.sym.M
    K = tkl.sym.K

    @tk.gen.thread(M)
    def rms_norm_kernel(
        input: tkl.OutputBuffer[M, K, dtype],
        weight: tk.lang.InputBuffer[M, K, dtype],
        output: tk.lang.OutputBuffer[M, K, dtype],
    ):
        row_index = tk.lang.program_id(0)
        eps = tkl.constant((1,), dtype, 0.00001)
        zero = tkl.constant((1,), dtype, 0.0)
        input_row = input[row_index, :]
        sq_inp = input_row * input_row
        sq_inp_red = tkl.sum(sq_inp)
        # TODO: The input_row * zero is just dummy computation to pass in the right shapes,
        # otherwise it leads to 'error: unknown: 'math.exp2' op operand #0 must be floating-point-like, but got 'vector<f16>'
        denom = tkl.rsqrt(input_row * zero + sq_inp_red)
        denom_eta = denom + eps
        output[row_index, :] = denom_eta * input_row * weight[row_index, :]

    with tk.gen.TestLaunchContext():
        rms_norm_kernel(input, weight, output)


def iota_krnl(dtype, input):
    M = tkl.sym.M

    @tk.gen.thread(M)
    def iota_kernel(out: tkl.OutputBuffer[M, dtype]):
        a = (
            tkl.constant((17, 37, 19), dtype, 5)
            if dtype in INT_DTYPES
            else tkl.constant((17, 37, 19), dtype, 5.0)
        )
        b = (
            tkl.constant((17, 37, 19), dtype, 10)
            if dtype in INT_DTYPES
            else tkl.constant((17, 37, 19), dtype, 10.0)
        )
        c = (
            tkl.constant((17, 37, 19), dtype, 2)
            if dtype in INT_DTYPES
            else tkl.constant((17, 37, 19), dtype, 2.0)
        )
        if dtype in INT_DTYPES:
            c = (a * b) // c
        else:
            c = (a * b) / c
        c = c + a - b

    with tk.gen.TestLaunchContext():
        iota_kernel(input)


def softmax_krnl(dtype, input, output):
    M = tkl.sym.M
    K = tkl.sym.K

    @tk.gen.thread(M)
    def softmax_kernel(
        input: tk.lang.InputBuffer[M, K, dtype],
        output: tk.lang.OutputBuffer[M, K, dtype],
    ):
        row_index = tk.lang.program_id(0)
        input_row = input[row_index, :]
        numerator = tkl.exp2(input_row - tkl.max(input_row))
        if dtype in INT_DTYPES:
            output_row = numerator // tkl.sum(numerator)
        else:
            output_row = numerator / tkl.sum(numerator)
        output[row_index, :] = output_row

    with tk.gen.TestLaunchContext():
        softmax_kernel(input, output)


def gemm_fx_kernel(dtype, A, B, output):
    N = tkl.sym.N
    M = tkl.sym.M
    K = tkl.sym.K
    BLOCK_SIZE = tkl.sym.BLOCK_SIZE

    @tk.gen.thread(N // BLOCK_SIZE, M // BLOCK_SIZE)
    def gemm_kernel(
        A: tkl.InputBuffer[N, K, dtype],
        B: tkl.InputBuffer[K, M, dtype],
        output: tkl.OutputBuffer[N, M, dtype],
    ):
        grid_n = tkl.program_id(0)
        grid_m = tkl.program_id(1)

        acc = None
        # TODO: Only considering the float and integer cases.
        if dtype in INT_DTYPES:
            acc = tkl.constant((BLOCK_SIZE, BLOCK_SIZE), dtype, 0)
        else:
            acc = tkl.constant((BLOCK_SIZE, BLOCK_SIZE), dtype, 0.0)

        @tkl.for_loop(0, K // BLOCK_SIZE, init_args=[acc])
        def body(i, c):
            a = tkl.load(A, (grid_n, i * BLOCK_SIZE), (BLOCK_SIZE, BLOCK_SIZE))
            b = tkl.load(B, (i * BLOCK_SIZE, grid_m), (BLOCK_SIZE, BLOCK_SIZE))
            return (tkl.dot(a, b, c),)

        tkl.store(output, (grid_n, grid_m), body[0])

    with tk.gen.TestLaunchContext({BLOCK_SIZE: 32}):
        gemm_kernel(A, B, output)


@pytest.mark.parametrize(
    ("dtype",),
    [(x,) for x in FLOAT_DTYPES + INT_DTYPES],
)
def test_iota_krnl(dtype):
    input = torch.zeros(17)
    iota_krnl(dtype, input)


@pytest.mark.parametrize(
    ("dtype",),
    [(x,) for x in FLOAT_DTYPES],
)
def test_rms_norm_krnl(dtype):
    input = torch.randn(128, 64).to(dtype.to_torch_type())
    weight = torch.randn(128, 64).to(dtype.to_torch_type())
    output = torch.randn(128, 64).to(dtype.to_torch_type())
    rms_norm_krnl(dtype, input, weight, output)


@pytest.mark.parametrize(
    ("dtype",),
    [(x,) for x in FLOAT_DTYPES],
)
def test_softmax_krnl(dtype):
    input = torch.randn(128, 64).to(dtype.to_torch_type())
    output = torch.randn(128, 64).to(dtype.to_torch_type())
    softmax_krnl(dtype, input, output)


@pytest.mark.parametrize(
    ("dtype",),
    [(x,) for x in FLOAT_DTYPES + INT_DTYPES],
)
def test_gemm_krnl(dtype):
    A = torch.randn(512, 1024).to(dtype.to_torch_type())
    B = torch.randn(1024, 2048).to(dtype.to_torch_type())
    output = torch.zeros(512, 2048).to(dtype.to_torch_type())
    gemm_fx_kernel(dtype, A, B, output)
