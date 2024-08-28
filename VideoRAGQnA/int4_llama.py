from ipex_llm import optimize_model
from transformers import LlamaForCausalLM
#import pdb
model = LlamaForCausalLM.from_pretrained('meta-llama/Llama-2-7b-chat-hf', torch_dtype='auto', low_cpu_mem_usage=True)

# Apply symmetric INT8 optimization
model = optimize_model(model, low_bit="sym_int4")

saved_dir='./llama-2-ipex-llm-4-bit'
model.save_low_bit(saved_dir)

#from ipex_llm.optimize import low_memory_init, load_low_bit
#with low_memory_init(): # Fast and low cost by loading model on meta device
#   model = LlamaForCausalLM.from_pretrained(saved_dir,
 #                                           torch_dtype="auto",
  #                                          trust_remote_code=True)
#pdb.set_trace()
#model = load_low_bit(model, saved_dir) # Load the optimized model
#print("Loaded optimized model")
