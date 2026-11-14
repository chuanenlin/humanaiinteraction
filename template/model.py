"""Everything that talks to a model lives in this file.

generate() returns a dictionary:
    text        the model's answer
    tokens      a list of {"text": ..., "prob": ...}, one per token, or None if the model
                does not report probabilities
    confidence  one number between 0 and 1 for the whole answer, or None

Pick the model with the MODEL_BACKEND environment variable:
    local  (default) a small open model that runs on your computer with Hugging Face transformers
    api    any provider with an OpenAI-compatible API (see generate_api below)
    fake   no model at all, for working on the front end quickly
"""

import math
import os
import random
import time

BACKEND = os.environ.get("MODEL_BACKEND", "local")
SYSTEM_PROMPT = "You are a helpful assistant. Keep answers short."


# ---------------------------------------------------------------------------
# Local model (default)
# ---------------------------------------------------------------------------

LOCAL_MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
_local = {}


def _load_local():
    if not _local:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            LOCAL_MODEL_ID, dtype=torch.float16 if device != "cpu" else torch.float32
        ).to(device)
        model.eval()
        _local.update(torch=torch, tokenizer=tokenizer, model=model, device=device)
        print(f"Loaded {LOCAL_MODEL_ID} on {device}")
    return _local


def generate_local(prompt, temperature=0.7, max_new_tokens=200):
    m = _load_local()
    torch, tokenizer, model = m["torch"], m["tokenizer"], m["model"]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(m["device"])

    settings = {"max_new_tokens": max_new_tokens, "pad_token_id": tokenizer.eos_token_id,
                "output_logits": True, "return_dict_in_generate": True}
    if temperature > 0:
        settings.update(do_sample=True, temperature=temperature, top_p=1.0, top_k=0)
    else:
        settings["do_sample"] = False

    with torch.no_grad():
        out = model.generate(**inputs, **settings)

    tokens = []
    new_ids = out.sequences[0, inputs["input_ids"].shape[1]:]
    for step, token_id in enumerate(new_ids):
        if int(token_id) == tokenizer.eos_token_id:
            break
        # The probability the model gave this token before sampling (temperature 1).
        prob = torch.softmax(out.logits[step][0].float(), dim=-1)[token_id].item()
        tokens.append({"text": tokenizer.decode([int(token_id)]), "prob": round(prob, 4)})

    probs = [t["prob"] for t in tokens]
    confidence = None
    if probs:
        # Geometric mean of the token probabilities: the average probability per token.
        log_sum = sum(math.log(max(p, 1e-12)) for p in probs)
        confidence = round(math.exp(log_sum / len(probs)), 4)
    return {"text": "".join(t["text"] for t in tokens).strip(), "tokens": tokens, "confidence": confidence}


# ---------------------------------------------------------------------------
# API model: SWAP IN ANY PROVIDER HERE
# ---------------------------------------------------------------------------
# Most providers accept the same request format (OpenAI's chat completions). To use one:
#   1. Get an API key from the provider (many have a free tier).
#   2. Set these environment variables before you start the app (never write the key in code):
#        MODEL_BACKEND=api
#        API_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/   (Google Gemini, for example)
#        API_MODEL=<a model name from the provider's documentation>
#        API_KEY=<your key>
#        API_REASONING_EFFORT=low   (only for a model that thinks before it answers, such as Gemini)
#   3. Restart the app.
# Most APIs do not return token probabilities by default, so tokens and confidence are None here.
# Some providers return them if you add "logprobs": True to the request; see their documentation.

def generate_api(prompt, temperature=0.7, max_new_tokens=200):
    import requests

    base_url = os.environ["API_BASE_URL"].rstrip("/") + "/"
    body = {
        "model": os.environ["API_MODEL"],
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        "temperature": temperature,
        # A model that thinks before it answers counts the thinking toward this limit,
        # so it is higher than max_new_tokens for the local model.
        "max_tokens": int(os.environ.get("API_MAX_TOKENS", 2000)),
    }
    if os.environ.get("API_REASONING_EFFORT"):
        body["reasoning_effort"] = os.environ["API_REASONING_EFFORT"]
    response = requests.post(
        base_url + "chat/completions",
        headers={"Authorization": f"Bearer {os.environ['API_KEY']}"},
        json=body,
        timeout=60,
    )
    response.raise_for_status()
    choice = response.json()["choices"][0]
    text = choice["message"].get("content")
    if not text:
        raise RuntimeError(f"the model sent back no text (finish_reason: {choice.get('finish_reason')}). "
                           "If it thinks before it answers, raise API_MAX_TOKENS or set API_REASONING_EFFORT=low.")
    return {"text": text.strip(), "tokens": None, "confidence": None}


# ---------------------------------------------------------------------------
# Fake model, for building the front end without waiting for a model
# ---------------------------------------------------------------------------

def generate_fake(prompt, temperature=0.7, max_new_tokens=200):
    time.sleep(0.3)
    words = f"This is a pretend answer to: {prompt}".split(" ")
    tokens = [{"text": (" " if i else "") + w, "prob": round(random.uniform(0.2, 1.0), 2)} for i, w in enumerate(words)]
    return {"text": " ".join(words), "tokens": tokens, "confidence": round(random.uniform(0.4, 0.95), 2)}


def generate(prompt, temperature=0.7, max_new_tokens=200):
    if BACKEND == "api":
        return generate_api(prompt, temperature, max_new_tokens)
    if BACKEND == "fake":
        return generate_fake(prompt, temperature, max_new_tokens)
    return generate_local(prompt, temperature, max_new_tokens)
