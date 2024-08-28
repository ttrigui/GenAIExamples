from ipex_llm import optimize_model
from transformers import LlamaForCausalLM

model = LlamaForCausalLM.from_pretrained('meta-llama/Llama-2-7b-chat-hf', torch_dtype='auto', low_cpu_mem_usage=True)

# Apply symmetric INT4 optimization
model = optimize_model(model, low_bit="sym_int4")

saved_dir='./llama-2-ipex-llm-4-bit'
model.save_low_bit(saved_dir)
