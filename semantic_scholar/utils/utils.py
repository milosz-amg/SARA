import re
import json

PERSONY_LABELS = {"PERSON", "PER", "persName", "orgName", "name"}

def extract_json_array(text: str):
    """Return the first JSON array found in text, robust to code fences/prose."""
    # strip common code fences
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    # if the whole text looks like an array, parse directly
    if text.lstrip().startswith("[") and text.rstrip().endswith("]"):
        return json.loads(text)
    # otherwise sniff the first [...] block
    m = re.search(r"\[[\s\S]*\]", text)
    if not m:
        raise json.JSONDecodeError("No JSON array found", text, 0)
    return json.loads(m.group(0))