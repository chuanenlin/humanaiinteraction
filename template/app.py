"""The web server. Run it with:  python app.py   then open http://127.0.0.1:8000

It does three things:
  - serves the front end in static/
  - POST /api/generate   sends a request to the model (see model.py) and returns its answer
  - POST /api/log        saves what the person did (edits, regenerations, ...) to logs/interactions.jsonl
"""

import json
import os
import time
import uuid

from flask import Flask, jsonify, request, send_from_directory

import model

app = Flask(__name__, static_folder="static")
LOG_FILE = os.path.join(os.path.dirname(__file__), "logs", "interactions.jsonl")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)


def write_log(event):
    """Append one event to the log file. You can read it later with pandas.read_json(LOG_FILE, lines=True)."""
    event["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(event) + "\n")


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.post("/api/generate")
def generate():
    data = request.get_json(force=True)
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "The request is empty."}), 400
    temperature = float(data.get("temperature", 0.7))

    started = time.time()
    try:
        result = model.generate(prompt, temperature=temperature)
    except Exception as error:  # show the problem in the page instead of failing silently
        return jsonify({"error": f"The model call failed: {error}"}), 500

    result["id"] = uuid.uuid4().hex[:8]
    result["seconds"] = round(time.time() - started, 2)
    write_log({"event": "generate", "session": data.get("session"), "id": result["id"], "prompt": prompt,
               "temperature": temperature, "text": result["text"], "confidence": result["confidence"]})
    return jsonify(result)


@app.post("/api/log")
def log_event():
    write_log(request.get_json(force=True))
    return jsonify({"ok": True})


if __name__ == "__main__":
    # Port 8000, not Flask's usual 5000: on a Mac, AirPlay Receiver answers on localhost:5000.
    port = int(os.environ.get("PORT", 8000))
    print(f"Model backend: {model.BACKEND}. Open http://127.0.0.1:{port} in your browser.")
    app.run(host="127.0.0.1", port=port, debug=False)
