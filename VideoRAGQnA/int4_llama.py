# Code from https://github.com/intel-analytics/ipex-llm/blob/main/docs/mddocs/Overview/KeyFeatures/optimize_model.md
from ipex_llm import optimize_model
from transformers import LlamaForCausalLM
import os

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
print(HF_TOKEN)
model = LlamaForCausalLM.from_pretrained('meta-llama/Llama-2-7b-chat-hf', torch_dtype='auto', low_cpu_mem_usage=True, token=HF_TOKEN)

# Apply symmetric INT4 optimization
model = optimize_model(model, low_bit="sym_int4")

saved_dir='./llama-2-ipex-llm-4-bit'
model.save_low_bit(saved_dir)
