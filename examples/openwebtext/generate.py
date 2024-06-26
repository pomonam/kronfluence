import torch
import transformers

model_id = "meta-llama/Meta-Llama-3-8B"
pipeline = transformers.pipeline(
    "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
)

# prompt = "Doctors are male and nurses are"
# prompt = "Machine learning can be defined as"
# prompt = "Using a distributed database has many advantages."

# prompt = """Story title: 3 llamas go for a walk
# Summary: The 3 llamas crossed a bridge and something unexpected happened
#
# Once upon a time"""

# prompt = "Inflation is typically measured by"
prompt = "Human: What is your name? Assistant:"
#there were 3 llamas: Llama 1, Llama 2 and Llama 3. They decided to go for a walk. They walked to a bridge and the first thing they saw was a sign that said: "Do not cross the bridge unless you have a very good reason to do so". They were curious so they asked the sign: "What's the reason to cross the bridge?"

outputs = pipeline(prompt, max_length=128)
print("Prompt:")
print(prompt)
print("Prompt + Completion:")
print(outputs[0]["generated_text"])
print(outputs[0]["generated_text"][len(prompt):])
