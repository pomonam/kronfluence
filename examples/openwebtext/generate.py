import torch
import transformers

model_id = "meta-llama/Meta-Llama-3-8B"

pipeline = transformers.pipeline(
    "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
)

prompt = "Three ways to stay healthy are"

outputs = pipeline(
    prompt,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
print(outputs[0]["generated_text"])
print(outputs[0]["generated_text"][len(prompt):])
