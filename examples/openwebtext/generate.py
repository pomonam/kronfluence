import torch
import transformers

model_id = "meta-llama/Meta-Llama-3-8B"
pipeline = transformers.pipeline(
    "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
)

prompt = "Doctors are male and nurses are"
# prompt = "Machine learning can be defined as"
# prompt = "Using a distributed database has many advantages."
# prompt = "Inflation is typically measured by"

outputs = pipeline(prompt)
print("Prompt:")
print(prompt)
print("Prompt + Completion:")
print(outputs[0]["generated_text"])
