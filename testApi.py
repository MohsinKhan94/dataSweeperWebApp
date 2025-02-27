import google.generativeai as genai

genai.configure(api_key="AIzaSyBSFiTjyh-DpvSYU-eYIhTLnuvqndrec3A")

# List available models
available_models = list(genai.list_models())

# Print model names
for model in available_models:
    print(model.name)
