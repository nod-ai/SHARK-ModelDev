import pytest
from turbine_models.turbine_tank import tank_util
from turbine_models.model_builder import HFTransformerBuilder
from turbine_models.model_runner import vmfbRunner
from turbine_models.custom_models.sd_inference import utils
from iree import runtime as ireert
import os


@pytest.mark.parametrize(
    "model_name,model_type,expected_err,run_e2e",
    [
        ("microsoft/resnet-50", "hf_img_cls", 8e-05, True),
        ("bert-large-uncased", "hf", 8e-06, True),
        ("facebook/deit-small-distilled-patch16-224", "hf_img_cls", 8e-05, True),
        ("google/vit-base-patch16-224", "hf_img_cls", 8e-05, True),
        ("microsoft/beit-base-patch16-224-pt22k-ft22k", "hf_img_cls", 8e-05, True),
        ("microsoft/MiniLM-L12-H384-uncased", "hf", 5e-07, True),
        ("google/mobilebert-uncased", "hf", 4.3, True),
        ("mobilenet_v3_small", "vision", 6e-05, True),
        ("nvidia/mit-b0", "hf_img_cls", 7.3, True),
        ("resnet101", "vision", 8e-06, True),
        ("resnet18", "vision", 8e-06, True),
        ("resnet50", "vision", 8e-06, True),
        ("squeezenet1_0", "vision", 9e-06, True),
        ("wide_resnet50_2", "vision", 9e-06, True),
        ("mnasnet1_0", "vision", 2e-05, True),
        pytest.param(
            "t5-base",
            "hf_seq2seq",
            -1,
            False,
            marks=pytest.mark.xfail(reason="iree-compile fails"),
        ),
        pytest.param(
            "t5-large",
            "hf_seq2seq",
            -1,
            False,
            marks=pytest.mark.xfail(reason="iree-compile fails"),
        ),
        ("openai/whisper-base", "hf_causallm", 9e-05, True),
        ("openai/whisper-small", "hf_causallm", 0.0003, True),
        ("openai/whisper-medium", "hf_causallm", 0.0003, True),
        ("facebook/opt-350m", "hf", 9e-07, True),
        ("facebook/opt-1.3b", "hf", 9e-06, True),
        ("BAAI/bge-base-en-v1.5", "hf", 9e-07, True),
        pytest.param(
            "facebook/bart-large",
            "hf_seq2seq",
            -1,
            False,
            marks=pytest.mark.xfail(reason="iree-compile fails"),
        ),
        pytest.param(
            "gpt2",
            "hf",
            -1,
            False,
            marks=pytest.mark.xfail(reason="iree-compile fails"),
        ),
        pytest.param(
            "gpt2-xl",
            "hf",
            -1,
            False,
            marks=pytest.mark.xfail(reason="iree-compile fails"),
        ),
        ("lmsys/vicuna-13b-v1.3", "hf", 5e-05, True),
        pytest.param(
            "microsoft/phi-1_5",
            "hf_causallm",
            -1,
            True,
            marks=pytest.mark.xfail(reason="correctness issue"),
        ),  # nan error reported (correctness issue)
        pytest.param(
            "microsoft/phi-2",
            "hf_causallm",
            -1,
            True,
            marks=pytest.mark.xfail(reason="correctness issue"),
        ),  # nan error reported (correctness issue)
        pytest.param(
            "mosaicml/mpt-30b",
            "hf_causallm",
            -1,
            False,
            marks=pytest.mark.xfail(reason="iree-compile fails"),
        ),
        ("stabilityai/stablelm-3b-4e1t", "hf_causallm", 0.0004, True),
    ],
)
def test_all_models(model_name, model_type, expected_err, run_e2e):
    import_args = {
        "batch_size": 1,
    }
    # Based on the model type, get the appropriate hugging face model, inputs, and output
    if model_type == "vision":
        torch_model, input, out = tank_util.get_vision_model(model_name, import_args)
    elif model_type == "hf":
        torch_model, input, out = tank_util.get_hf_model(model_name, import_args)
    elif model_type == "hf_seq2seq":
        torch_model, input, out = tank_util.get_hf_seq2seq_model(
            model_name, import_args
        )
    elif model_type == "hf_causallm":
        torch_model, input, out = tank_util.get_hf_causallm_model(
            model_name, import_args
        )
    elif model_type == "hf_img_cls":
        torch_model, input, out = tank_util.get_hf_img_cls_model(
            model_name, import_args
        )
    # compile model and get vmfb
    model = HFTransformerBuilder(
        example_input=input,
        hf_id=model_name,
        hf_auth_token="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",
        upload_ir=True,
        model=torch_model,
        model_type=model_type,
        run_e2e=run_e2e,
    )
    vmfb_name = model_name.replace("/", "_") + ".vmfb"
    model.get_compiled_module(save_to=vmfb_name)

    # if model is not supposed to run e2e, exit at this point (mlir has been uploaded)
    if run_e2e is False:
        assert expected_err > 0
        return

    # run inference using iree runtime
    runner = vmfbRunner("local-task", vmfb_name)
    inputs = [ireert.asdevicearray(runner.config.device, input)]
    keys = list(runner.ctx.modules)
    key = keys[len(keys) - 1]
    results = runner.ctx.modules.__getattr__(key)["main"](*inputs)
    err = utils.largest_error(out.cpu().detach().numpy(), results)
    # cleanup
    os.remove(vmfb_name)
    # accuracy
    assert err < expected_err
