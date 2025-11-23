# backend/agents/gemini_list_and_test.py
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

genai.configure(api_key=key)

# 1) List available models
print("Listing models visible to this key...\n")
try:
    models = genai.list_models()
except Exception as e:
    print("ERR_LIST_MODELS:", e)
    sys.exit(3)

# models may be a dict-like or have .models attribute; handle both
model_entries = []
if isinstance(models, dict) and "models" in models:
    model_entries = models["models"]
else:
    # try iterable
    try:
        model_entries = list(models)
    except Exception:
        model_entries = [models]

# Print a short summary of each model
for m in model_entries:
    # m may be dict-like or object; try various fields
    mid = getattr(m, "name", None) or m.get("name") if isinstance(m, dict) else str(m)
    mtype = getattr(m, "type", None) or m.get("type") if isinstance(m, dict) else None
    # If supported methods available, print them
    methods = None
    if isinstance(m, dict):
        methods = m.get("supported_methods") or m.get("capabilities") or m.get("features")
    else:
        methods = getattr(m, "supported_methods", None)
    print("MODEL ID:", mid)
    if methods:
        print("  supported_methods:", methods)
    print()

# 2) If user set GEMINI_MODEL, try generate with it; otherwise pick a candidate
chosen = os.getenv("GEMINI_MODEL")
if not chosen:
    # pick first model id that likely supports generation
    for m in model_entries:
        mid = getattr(m, "name", None) or (m.get("name") if isinstance(m, dict) else None)
        if not mid:
            continue
        low = mid.lower()
        if "gemini" in low or "bison" in low or "chat" in low or "text" in low:
            chosen = mid
            break

if not chosen:
    print("NO_SUGGESTED_MODEL_FOUND. Please inspect the printed model IDs and set GEMINI_MODEL to a model that supports generation.")
    sys.exit(0)

print("Attempting a short generate with model:", chosen)
try:
    # create a generative model object and call generate_content
    m = genai.GenerativeModel(chosen)
    resp = m.generate_content("Say hi in one short sentence.")
    print("OK:", resp.text.strip()[:400])
except Exception as e:
    print("ERR_GENERATE:", e)
    sys.exit(4)
