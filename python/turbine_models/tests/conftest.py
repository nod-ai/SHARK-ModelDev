def pytest_addoption(parser):
    parser.addoption("--all", action="store_true", help="run all combinations")


def pytest_generate_tests(metafunc):
    if "quantization" in metafunc.fixturenames:
        if metafunc.config.getoption("all"):
            quantizations = ["int4", None]
        else:
            quantizations = ["int4"]
        metafunc.parametrize("quantization", quantizations)
    
    if "precision" in metafunc.fixturenames:
        if metafunc.config.getoption("all"):
            precisions = ["f16", "f32"]
        else:
            precisions = ["f16"]
        metafunc.parametrize("precision", precisions)