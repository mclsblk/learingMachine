import torch
import transformers
from transformers import pipeline
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from transformers import AutoTokenizer

torch.cuda.empty_cache()

model_id = "D:\miniconda3\_llama3.2_1B_add"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = transformers.AutoModelForCausalLM.from_pretrained(model_id)
pipe = pipeline(
    "text-generation", 
    model=model, 
    torch_dtype=torch.bfloat16, 
    tokenizer=tokenizer,
    device=0,
    max_new_tokens=50
)


# model3 = "D:\miniconda3\_deepseek_distil_llama_8B_add"
# token3 = AutoTokenizer.from_pretrained(model3)
# model3 = transformers.AutoModelForCausalLM.from_pretrained(model3)
# pipe = pipeline(
#     "text-generation", 
#     model=model3, 
#     torch_dtype=torch.int8, 
#     device="cuda:0",
#     tokenizer=token3
# )

print(pipe("The key to life is"))