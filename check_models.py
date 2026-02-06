import google.generativeai as genai
import os

# Ask for the key securely
key = input("Enter API Key: ").strip()
genai.configure(api_key=key)

print("\nScanning available models...")
found = False
for m in genai.list_models():
    if 'embedContent' in m.supported_generation_methods:
        print(f"AVAILABLE: {m.name}")
        found = True

if not found:
    print("NO EMBEDDING MODELS FOUND. Ensure 'Generative Language API' is enabled in Google Console.")
