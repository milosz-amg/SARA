import json
import re
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

def get_api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    return api_key

def get_client():
    return OpenAI(api_key=get_api_key())

def call_openai_json(prompt: str, model="gpt-4o", temperature=1.0, use_web_search=False) -> dict:
    client = get_client()
    
    if use_web_search and model == "gpt-4o":
        print("[INFO] Using web search for current information")
        return _call_with_web_search(client, prompt)
    else:
        print("[INFO] Using standard chat completion")
        return _call_standard_chat(client, prompt, model, temperature)

def _call_with_web_search(client, prompt: str) -> dict:
    try:
        response = client.responses.create(
            model="gpt-4o",
            input=prompt,
            tools=[{"type": "web_search"}]
        )
        
        output_message = next(
            (item for item in response.output if getattr(item, "type", "") == "message"), None
        )
        
        if not output_message:
            print("[ERROR] No assistant message found.")
            return {"error": "No output message found"}
        
        texts = [c.text for c in getattr(output_message, "content", []) if hasattr(c, "text")]
        content = "\n".join(texts).strip()
        print(f"[INFO] Raw content (web search): {content}")
        
        try:
            texts = [c.text for c in getattr(output_message, "content", []) if hasattr(c, "text")]
            content = "\n".join(texts).strip()
            # print(f"[INFO] Raw content (web search): {content}")

            if content.startswith("{") or content.startswith("["):
                return json.loads(content)
            else:
                return {"response": content}

        except json.JSONDecodeError as e:
            print(f"JSON decode failed: {e}")
            return {"response": content}
            
    except Exception as e:
        print(f"[ERROR] Web search API error: {e}")
        return {"error": str(e)}

def _call_standard_chat(client, prompt: str, model: str, temperature: float) -> dict:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        
        raw = response.choices[0].message.content.strip()
        
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
        
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            print("Failed to parse JSON. Returning raw response.")
            print("Failed to parse JSON. Raw response:")
            print(raw)
            return {"response": raw}
            
    except Exception as e:
        print(f"[ERROR] Standard chat API error: {e}")
        return {"error": str(e)}
    
call_openai_json("Kim jest Patryk Żywica. Pracuje na UAM", use_web_search=True)
#call_openai_json("Kim jest Patryk Żywica. Pracuje na UAM", use_web_search=False)