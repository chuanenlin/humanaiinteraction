// The front end: send the request, show the answer with its confidence, and let the
// person regenerate, edit, and go back to earlier versions. Every action is logged.

const session = Math.random().toString(36).slice(2, 10); // one id per page load, to group a person's actions
const versions = [];   // every answer shown so far: {id, text, tokens, confidence, edited}
let current = -1;      // index of the version on screen
let lastPrompt = "";

const UNSURE_BELOW = 0.3; // highlight tokens the model gave less than this probability
const $ = (id) => document.getElementById(id);

function log(event, details = {}) {
  fetch("/api/log", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ event, session, ...details }),
  });
}

async function ask(prompt) {
  $("status").textContent = "Thinking...";
  $("send").disabled = $("regenerate").disabled = true;
  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, temperature: Number($("temperature").value), session }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error);
    versions.push({ ...data, edited: false });
    show(versions.length - 1);
    $("status").textContent = `${data.seconds} s`;
  } catch (error) {
    $("status").textContent = error.message;
  } finally {
    $("send").disabled = $("regenerate").disabled = false;
  }
}

// How sure the model was, in words and as a number. The cutoffs are a design choice:
// check them against how often the model is actually right on your task (Lecture 8).
function describeConfidence(c) {
  if (c === null || c === undefined) return "Confidence not available for this model";
  const pct = Math.round(c * 100);
  if (c >= 0.9) return `Fairly sure (${pct}%)`;
  if (c >= 0.6) return `Somewhat sure (${pct}%)`;
  return `Not sure (${pct}%)`;
}

function show(index) {
  current = index;
  const v = versions[index];
  $("result").hidden = false;
  $("confidence").textContent = describeConfidence(v.confidence);
  $("confidence").dataset.level = v.confidence == null ? "none" : v.confidence >= 0.9 ? "high" : v.confidence >= 0.6 ? "mid" : "low";

  const answer = $("answer");
  answer.contentEditable = "false";
  answer.textContent = "";
  if (v.tokens && !v.edited && $("highlight").checked) {
    for (const t of v.tokens) {
      const span = document.createElement("span");
      span.textContent = t.text;
      if (t.prob < UNSURE_BELOW) {
        span.className = "unsure";
        span.style.setProperty("--alpha", (1 - t.prob).toFixed(2));
      }
      span.title = `${Math.round(t.prob * 100)}%`;
      answer.appendChild(span);
    }
  } else {
    answer.textContent = v.text;
  }
  $("edit").hidden = false;
  $("keep").hidden = true;
  renderHistory();
}

function renderHistory() {
  const list = $("history");
  list.textContent = "";
  versions.forEach((v, i) => {
    const item = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.className = i === current ? "current" : "";
    const label = v.edited ? "Your edit" : v.confidence == null ? "Model" : `Model, ${Math.round(v.confidence * 100)}%`;
    button.textContent = `${label}: ` + v.text.slice(0, 70) + (v.text.length > 70 ? "..." : "");
    button.onclick = () => { show(i); log("go_back", { to: v.id }); };
    item.appendChild(button);
    list.appendChild(item);
  });
}

$("ask").addEventListener("submit", (e) => {
  e.preventDefault();
  lastPrompt = $("prompt").value.trim();
  if (lastPrompt) ask(lastPrompt);
});

$("regenerate").onclick = () => {
  log("regenerate", { from: versions[current].id });
  ask(lastPrompt);
};

$("edit").onclick = () => {
  const answer = $("answer");
  answer.textContent = versions[current].text; // drop the highlighting while editing
  answer.contentEditable = "true";
  answer.focus();
  $("edit").hidden = true;
  $("keep").hidden = false;
};

$("keep").onclick = () => {
  const before = versions[current];
  const text = $("answer").innerText.trim();
  versions.push({ id: before.id + "-edit", text, tokens: null, confidence: before.confidence, edited: true });
  log("edit", { from: before.id, before: before.text, after: text });
  show(versions.length - 1);
};

$("highlight").onchange = () => { if (current >= 0) show(current); };
$("temperature").oninput = () => { $("temperature-value").textContent = $("temperature").value; };
