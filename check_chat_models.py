import google.generativeai as genai
import os

key = input("Enter API Key: ").strip()
genai.configure(api_key=key)

print("\n🔍 Scanning for CHAT models...")
found = False
for m in genai.list_models():
    # specifically looking for 'generateContent' (Chat) ability
    if 'generateContent' in m.supported_generation_methods:
        print(f"✅ AVAILABLE: {m.name}")
        found = True

if not found:
    print("❌ No chat models found.")
