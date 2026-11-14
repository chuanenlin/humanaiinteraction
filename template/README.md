# Lab 3: Prototype AI features in a web app

This is a small web app you can build your project's AI features in. It goes with Lecture 12 and Milestone 3, and you can keep using it for Milestone 4. A Python backend (Flask) calls a model and a plain HTML and JavaScript page shows the answer. There is no build step or framework to learn.

The example feature shows the model's answer with how sure the model was and highlights the words it was least sure about. A person can also regenerate the answer, edit it, and go back to an earlier version. Every request, regeneration and edit is saved to `logs/interactions.jsonl`, so you can use the log as data in your user tests.

## Run it

You need Python 3.10 or newer. In a terminal, in this folder:

```
python -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:8000 in your browser. The first request downloads the model (about 1 GB), so it takes a minute or two, and after that an answer takes a few seconds on a laptop.

If you want to work on the page without waiting for a model, you can start it with a pretend model that answers right away:

```
MODEL_BACKEND=fake python app.py           # on Windows (PowerShell): $env:MODEL_BACKEND="fake"; python app.py
```

If another program already uses port 8000, you can pick another port, for example `PORT=8001 python app.py`. On a Mac, do not use 5000, since AirPlay Receiver answers on that port.

## What is in each file

| file | what it does |
|---|---|
| `app.py` | The web server. It serves the page, sends requests to the model (`/api/generate`), and writes the log (`/api/log`). |
| `model.py` | Everything that talks to a model. `generate()` returns the answer, the probability of each token, and one confidence number. |
| `static/index.html` | The page. |
| `static/app.js` | What the page does: asking, showing the confidence, highlighting, regenerating, editing, and the list of versions. |
| `static/style.css` | How the page looks. |
| `logs/interactions.jsonl` | Created when you first use the app, with one line per event. |

## Change the model

The default is Qwen2.5-0.5B-Instruct, a small open model that runs on most laptops. It is fast but often wrong, so if your computer has enough memory (about 8 GB free), you can use a larger open model for better answers:

```
MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct python app.py
```

To use a model behind an API instead, see `generate_api` in `model.py`, which is marked "SWAP IN ANY PROVIDER HERE". Most providers accept the same request format, so you only need to set these four environment variables and restart:

```
MODEL_BACKEND=api
API_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
API_MODEL=<a model name from the provider's documentation>
API_KEY=<your key>
```

The example address is Google's Gemini API, which has a free tier. Please never write a key into the code or commit it to GitHub. Most APIs do not send token probabilities back, so the page will say that confidence is not available. If your provider can return them, you can ask for them in `generate_api` and fill in `tokens` and `confidence`.

## About the confidence number

The number is the average probability per token (the geometric mean), and you should not take it at face value (Lecture 8). A small model gives low probabilities to many words in open-ended writing, because there are many fine ways to word the same sentence. The labels "Fairly sure", "Somewhat sure" and "Not sure" in `describeConfidence` (in `app.js`) are placeholders, so before you show them to anyone, check them against how often the model is actually right on your task.

## Use the log in your tests

Each line of `logs/interactions.jsonl` is one event (`generate`, `regenerate`, `edit` or `go_back`) with the time and a session id for each page load. For example, to count regenerations and edits per person:

```python
import pandas as pd

log = pd.read_json("logs/interactions.jsonl", lines=True)
print(log.groupby(["session", "event"]).size().unstack(fill_value=0))
```

Please tell the people you test with that the app saves what they type, and delete the log when the term ends.

## Add your own feature

- You can change the page in `static/index.html` and `static/app.js`. To log something new, call `log("event_name", { ...details })` in `app.js`.
- When a feature needs more than one model call, you can add a route in `app.py`. For example, this small chain (Lecture 12) first asks for steps and then does each one:

```python
@app.post("/api/chain")
def chain():
    task = request.get_json(force=True)["prompt"]
    steps = model.generate(f"List at most 4 steps to do this, one per line, with no other text: {task}")["text"]
    steps = [s for s in steps.splitlines() if s.strip()]
    results = [model.generate(f"Task: {task}\nDo only this step: {s}")["text"] for s in steps]
    return jsonify({"steps": steps, "results": results})
```

- For an agent that asks before it acts (Milestone 4), you can copy the `POLICY` table and the `run_agent` loop from the Lab 5 notebook into a route like the one above, then show the plan in the page with a button for each step.

## License

MIT, like the rest of the course materials. See `LICENSE`.
