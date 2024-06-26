import torch
import transformers

model_id = "meta-llama/Meta-Llama-3-8B"

pipeline = transformers.pipeline(
    "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
)

prompt = "Doctors are male and nurses are "

outputs = pipeline(prompt)
print(outputs[0]["generated_text"])
print(outputs[0]["generated_text"][len(prompt):])
