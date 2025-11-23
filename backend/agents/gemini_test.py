# backend\agents\gemini_test.py
import os, sys
try:
    import google.generativeai as genai
except Exception as e:
    print("IMPORT_ERR", e)
    sys.exit(2)

key = os.getenv("GEMINI_API_KEY")
if not key:
    print("NO_KEY")
    sys.exit(1)

try:
    genai.configure(api_key=key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    resp = model.generate_content("Say hi in one short sentence.")
    print("OK:", resp.text.strip()[:300])
except Exception as e:
    print("ERR:", e)
    sys.exit(3)
