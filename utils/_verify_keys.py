import json, re, sys
sys.stdout.reconfigure(encoding="utf-8")

with open("ChaviOllama.json", encoding="utf-8") as f:
    d = json.load(f)
keys = d["CLOUD_API_KEYS"]
print(f"ChaviOllama.json: {len(keys)} voci")
for k in keys:
    print(f"  {k['name']}")

print()
with open("ask_llm_multiple_CoT.py", encoding="utf-8") as f:
    src = f.read()
m = re.search(r"CLOUD_API_KEYS\s*=\s*\[(.*?)\]", src, re.DOTALL)
names = re.findall(r"#\s*(\S+)", m.group(1))
print(f"CLOUD_API_KEYS nel .py: {len(names)} chiavi")
for n in names:
    print(f"  {n}")
