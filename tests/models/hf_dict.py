import torch

class ModelData():
    def __init__(self, input_shape, torch_dtype: torch.dtype, xfail: bool = True):
        self.xfail = xfail
        self.input_shape = input_shape
        self.dtype = torch_dtype

model_dict = {
    'distilgpt2': ModelData(input_shape=(1, 1), torch_dtype=torch.int64),
    'gpt2': ModelData(input_shape=(1, 1), torch_dtype=torch.int64),
    'gpt2-medium': ModelData(input_shape=(1, 1), torch_dtype=torch.int64),
    'bert-base-uncased': ModelData(input_shape=(1, 1), torch_dtype=torch.int64),
    'bert-large-uncased': ModelData(input_shape=(1, 1), torch_dtype=torch.int64),
}