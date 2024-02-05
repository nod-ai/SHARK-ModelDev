from typing import Any, List, Optional
import argparse
from turbine_models.model_runner import vmfbRunner
from transformers import AutoTokenizer
from iree import runtime as ireert
import torch
import time
from turbine_models.custom_models.llm_optimizations.streaming_llm.modify_llama import (
    enable_llama_pos_shift_attention,
)
from llama_index import SimpleDirectoryReader, VectorStoreIndex, ServiceContext, StorageContext, load_index_from_storage, PromptHelper
from llama_index.text_splitter import TokenTextSplitter, SentenceSplitter
from llama_index.node_parser import SimpleNodeParser
from llama_index.callbacks.base import CallbackManager
from llama_index.bridge.pydantic import PrivateAttr

parser = argparse.ArgumentParser()

# TODO move common runner flags to generic flag file
parser.add_argument(
    "--vmfb_path", type=str, default="", help="comma separated path to vmfbs"
)
parser.add_argument(
    "--external_weight_path",
    type=str,
    default="",
    help="path to external weight parameters if model compiled without them",
)
parser.add_argument(
    "--compare_vs_torch",
    action="store_true",
    help="Runs both turbine vmfb and a torch model to compare results",
)
parser.add_argument(
    "--hf_model_name",
    type=str,
    help="comma separated HF model name's",
    default="BAAI/bge-base-en-v1.5,microsoft/phi-2",
)
parser.add_argument(
    "--hf_auth_token",
    type=str,
    help="The Hugging face auth token, required for some models",
)
parser.add_argument(
    "--device",
    type=str,
    default="local-task",
    help="local-sync, local-task, cuda, vulkan, rocm",
)
parser.add_argument(
    "--data_dir",
    type=str,
    default="",
    help="Path to directory containing data for rag pipeline model",
)

parser.add_argument(
    "--prompt",
    type=str,
    default="What type of gas do I need to use?",
    help="Prompt for pipeline",
)

from llama_index.embeddings.base import BaseEmbedding
class vmfbEmbeddings(BaseEmbedding):
    _runner: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()
    def __init__(
        self,
        embedding_runner,
        tokenizer,
        callback_manager: Optional[CallbackManager] = None,
    ):
        self._runner = embedding_runner
        self._tokenizer = tokenizer
        embed_batch_size=1
        super().__init__(
            callback_manager=callback_manager,
            embed_batch_size=embed_batch_size,
            #runner = runner
        )
    def format_out(self, results):
        output = []
        for x in results:
            output.append(torch.tensor(x.to_host()[0][0]))
        return output#torch.tensor(results.to_host()[0][0]]
    
    def _get_query_embedding(self, query: str) -> List[float]:
        init_input = self._tokenizer(query, return_tensors="pt")
        text_input = [ireert.asdevicearray(self._runner.config.device, init_input.input_ids)]
        result = self._runner.ctx.modules.bge_model["run_embedding"](*text_input)
        output = self.format_out(result)
        return [output[-1].item()]

    def _aget_query_embedding(self, query: str) -> List[float]:
        print("async query embedding not implmented", flush=True)
        exit()
        return [1.0]
    
    def _get_text_embedding(self, text: str) -> List[float]:
        print("_get_text_embedding", flush=True)
        embeddings = self._runner.ctx.modules.bge_model["run_embedding"](text)
        return embeddings[0]
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        init_input = self._tokenizer(texts, return_tensors="pt")
        text_input = [ireert.asdevicearray(self._runner.config.device, init_input.input_ids)]
        out = self._runner.ctx.modules.bge_model["run_embedding"](*text_input)
        output = self.format_out(out)
        results = []
        results.append([output[-1].item()])
        return results

from llama_index.llms import CustomLLM, LLMMetadata, CompletionResponse, CompletionResponseGen
from llama_index.llms.base import llm_completion_callback
from llama_index.types import PydanticProgramMode
from llama_index.constants import DEFAULT_CONTEXT_WINDOW
class vmfbLlm(CustomLLM):
    _runner: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()    
    def __init__(
        self,
        runner,
        tokenizer_model_name,
        context_window: int = 4096,
        system_prompt: str = "",
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
    ):
        self._runner = runner
        num_output: int = 65
        model_name: str = "custom"
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name, use_fast=False)
        system_prompt = system_prompt
        context_window = context_window
        super().__init__(
            context_window=context_window,
            system_prompt=system_prompt,
            pydantic_program_mode=pydantic_program_mode,
        )
    
    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(context_window=4096, num_output=65, model_name="custom")

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        print(prompt)
        init_input = self._tokenizer(prompt, return_tensors="pt")
        inputs = [ireert.asdevicearray(self._runner.config.device, init_input.input_ids)]
        results = self._runner.ctx.modules.state_update["run_initialize"](*inputs)

        def llm_format_out(results):
            return torch.tensor(results.to_host()[0][0])

        output = []
        output.append(format_out(results))

        while format_out(results) != 2:
            results = self._runner.ctx.modules.state_update["run_forward"](results) 
            print(f"phi: {self._tokenizer.decode(self.llm_format_out(results))}")
            output.append(self.llm_format_out(results))
        return CompletionResonse(text=self._tokenizer.decode(output))

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        print("TODO: Streaming not implemented here")
        return CompletionResponse(text="not done")

def run_rag(
    device,
    data_dir,
    vmfb_path,
    hf_model_name,
    hf_auth_token,
    external_weight_path,
    prompt,
):
    hf_model_names = hf_model_name.split(",")
    vmfb_paths = vmfb_path.split(",") 
    external_weight_paths = external_weight_path.split(",") 
    print("parsing data")
    doc = SimpleDirectoryReader(input_dir=data_dir).load_data()
    embed_tokenizer = AutoTokenizer.from_pretrained(
        hf_model_names[0],
        use_fast=False,
        token=hf_auth_token,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        hf_model_names[1],
        use_fast=False,
        token=hf_auth_token,
    )      

    embedding_runner = vmfbRunner(device=device, vmfb_path=vmfb_paths[0], external_weight_path=external_weight_paths[0])
    embed_model = vmfbEmbeddings(embedding_runner, embed_tokenizer)
    llm_runner = vmfbRunner(device=device, vmfb_path=vmfb_paths[1], external_weight_path=external_weight_paths[1])
    llm_model = vmfbLlm(llm_runner, hf_model_names[1])

    service_context = ServiceContext.from_defaults(llm=llm_model, embed_model=embed_model)
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    index = VectorStoreIndex.from_documents(doc, service_context=service_context, storage_context=storage_context)
    index.storage_context.persist()
    #index = load_index_from_storage(storage_context)
    query_engine = index.as_query_engine(service_context=service_context)
    response = query_engine.query(prompt)
    return response
  
#TODO: implement torch rag comparison
def run_torch_rag(hf_model_name, hf_auth_token, prompt):
    hf_model_names = hf_model_name.split(",")    
    embed_model = "local:" + hf_model_names[0]
    llm_model = "local:" + hf_model_names[1]

    doc = SimpleDirectoryReader(input_dir=data_dir).load_data()
    
    service_context = ServiceContext.from_defaults(llm=llm_model, embed_model=embed_model) 
    index = VectorStoreIndex.from_documents(doc, service_context=service_context)

    query_engine = index.as_query_engine(service_context=service_context)
    response = query_engine.query(prompt)
    return response

if __name__ == "__main__":
    args = parser.parse_args()
    print("generating turbine output: ")
    turbine_output = run_rag(
        args.device,
        args.data_dir,
        args.vmfb_path,
        args.hf_model_name,
        args.hf_auth_token,
        args.external_weight_path,
        args.prompt,
    )
    print(turbine_output)
    if args.compare_vs_torch:
        print("generating torch output: ")
        torch_output = run_torch_llm(
            args.hf_model_name, args.prompt,
        )
        print(torch_output)
