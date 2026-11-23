import os
import re
import html
import requests
import pandas as pd
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from rag import process_and_store_excel, search_relevant_chunks

app = FastAPI()
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
DATA_FILE = "data/campus_data.xlsx"
DF = pd.DataFrame()


# ---------------------------------------------------------------- startup
@app.on_event("startup")
async def load_dataset():
    global DF
    try:
        chunks_path = os.path.join("vector_store", "chunks.pkl")
        if os.path.exists(chunks_path):
            os.remove(chunks_path)
        n = process_and_store_excel(DATA_FILE, "Campus Dataset", "General")
        print(f"Loaded {n} rows from dataset")
        DF = pd.read_excel(DATA_FILE).dropna(how="all").fillna("")
    except Exception as e:
        print("Dataset load failed:", e)


@app.get("/health")
async def health():
    return {"status": "ok", "rows": int(len(DF))}


# ---------------------------------------------------------------- layout
def page(title: str, active: str, body: str) -> str:
    def link(href, label, key):
        cls = "text-white bg-gray-800" if key == active else "text-gray-400 hover:text-white"
        return f'<a href="{href}" class="px-3 py-2 rounded-lg text-sm font-medium transition {cls}">{label}</a>'

    nav = (
        link("/chat", "Chat", "chat")
        + link("/dataset", "Dataset", "dataset")
        + link("/how-it-works", "How it works", "how")
    )
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>__TITLE__ — CampusNova</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://unpkg.com/htmx.org@1.9.12"></script>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    body { font-family: 'Inter', sans-serif; }
    .htmx-indicator { display: none; }
    .htmx-request .htmx-indicator, .htmx-request.htmx-indicator { display: flex; }
    .answer ul { list-style: disc; padding-left: 1.25rem; margin: .4rem 0; }
    .answer p { margin: .35rem 0; }
  </style>
</head>
<body class="bg-gray-950 text-white min-h-screen flex flex-col">
  <header class="border-b border-gray-800 bg-black sticky top-0 z-50">
    <div class="max-w-6xl mx-auto px-4 py-3 flex items-center justify-between gap-4">
      <a href="/" class="flex items-center gap-3">
        <div class="w-9 h-9 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-full flex items-center justify-center font-bold text-sm">CN</div>
        <div class="leading-tight">
          <div class="font-bold">CampusNova</div>
          <div class="text-xs text-gray-500">AI College Assistant</div>
        </div>
      </a>
      <nav class="flex items-center gap-1">__NAV__</nav>
    </div>
  </header>
  <main class="flex-1 w-full">__BODY__</main>
  <footer class="border-t border-gray-800 py-4 text-center text-xs text-gray-600">
    Mini Project · RAG + LLM over a college dataset · FastAPI · HTMX · TailwindCSS
  </footer>
</body>
</html>""".replace("__TITLE__", title).replace("__NAV__", nav).replace("__BODY__", body)


# ---------------------------------------------------------------- helpers
def to_html(text: str) -> str:
    """Safely turn the LLM's simple markdown (bold, bullets) into HTML."""
    text = html.escape(text.strip())
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    out, in_list = [], False
    for line in text.splitlines():
        s = line.strip()
        if re.match(r"^[-*•]\s+", s):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append("<li>" + re.sub(r"^[-*•]\s+", "", s) + "</li>")
        else:
            if in_list:
                out.append("</ul>")
                in_list = False
            if s:
                out.append(f"<p>{s}</p>")
    if in_list:
        out.append("</ul>")
    return "".join(out)


def source_chips(chunks) -> str:
    chips, seen = "", set()
    for c in chunks:
        m = re.match(r"\[Row (\d+)\]\s*Category:\s*([^|]*)\|\s*Topic:\s*([^|]*)", c["text"])
        label = f"{m.group(3).strip()} · Row {m.group(1)}" if m else c.get("doc_name", "Dataset")
        if label not in seen:
            seen.add(label)
            chips += (
                '<span class="bg-teal-900 text-teal-300 text-xs px-2 py-1 rounded-full">'
                + html.escape(label) + "</span>"
            )
    return chips


def bot_card(inner: str, badge: str = "From dataset", badge_cls: str = "text-green-400 bg-green-900") -> str:
    return f"""
    <div class="flex items-start gap-3">
      <div class="w-8 h-8 shrink-0 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-full flex items-center justify-center text-xs font-bold">CN</div>
      <div class="bg-gray-900 border border-gray-800 rounded-2xl rounded-tl-sm px-4 py-3 max-w-[85%]">
        <div class="mb-1"><span class="text-xs {badge_cls} px-2 py-0.5 rounded-full">{badge}</span></div>
        {inner}
      </div>
    </div>"""


# ---------------------------------------------------------------- pages
@app.get("/", response_class=HTMLResponse)
async def home():
    with open("templates/landing.html", encoding="utf-8") as f:
        return HTMLResponse(f.read())


SAMPLES = [
    ("💰", "What is the B.Tech tuition fee?"),
    ("🏠", "What is the hostel fee?"),
    ("🌙", "When does the girls hostel gate close?"),
    ("📅", "What attendance do I need to write exams?"),
    ("🎓", "How do I apply for a scholarship?"),
    ("💼", "Which companies come for placements?"),
    ("📈", "What is the average package?"),
    ("📄", "What documents are needed for admission?"),
    ("🚌", "Is there a bus facility?"),
    ("🔁", "How can I apply for revaluation?"),
]


@app.get("/chat", response_class=HTMLResponse)
async def chat_page():
    buttons = "".join(
        f'<button type="button" onclick="ask(this.dataset.q)" data-q="{html.escape(q)}" '
        f'class="w-full text-left text-sm text-gray-300 hover:text-white bg-gray-900 hover:bg-gray-800 '
        f'border border-gray-800 rounded-xl px-3 py-2 transition">{icon} {html.escape(q)}</button>'
        for icon, q in SAMPLES
    )
    welcome = bot_card(
        '<p class="text-sm text-gray-200">Hi! I am CampusNova. Ask me about fees, hostel, admission, '
        'exams, scholarships or placements. I answer only from the college dataset.</p>',
        "Ready", "text-indigo-300 bg-indigo-950",
    )
    body = """
<div class="max-w-6xl mx-auto px-4 py-6 grid grid-cols-1 lg:grid-cols-4 gap-6">
  <aside class="lg:col-span-1 order-2 lg:order-1">
    <h2 class="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-3">Try asking</h2>
    <div class="flex flex-col gap-2">__BUTTONS__</div>
    <button type="button" onclick="clearChat()" class="mt-4 w-full text-sm text-gray-400 hover:text-white border border-gray-800 rounded-xl px-3 py-2 transition">🗑 Clear chat</button>
  </aside>

  <section class="lg:col-span-3 order-1 lg:order-2 flex flex-col bg-black border border-gray-800 rounded-2xl overflow-hidden" style="height: 72vh;">
    <div id="messages" class="flex-1 overflow-y-auto p-4 flex flex-col gap-4">__WELCOME__</div>
    <div id="thinking" class="htmx-indicator px-4 pb-2 text-sm text-gray-500 items-center gap-2">
      <span class="w-2 h-2 bg-indigo-400 rounded-full animate-pulse"></span> CampusNova is searching the dataset and thinking…
    </div>
    <form id="chat-form" hx-post="/chat" hx-target="#messages" hx-swap="beforeend" hx-indicator="#thinking"
          hx-on::before-request="addUser()" hx-on::after-request="this.reset(); scrollDown()"
          class="border-t border-gray-800 p-3 flex gap-2">
      <input id="q" name="question" required autocomplete="off" placeholder="Ask about fees, hostel, admission, exams…"
             class="flex-1 bg-gray-900 border border-gray-700 focus:border-indigo-500 outline-none rounded-xl px-4 py-3 text-sm">
      <button class="bg-indigo-600 hover:bg-indigo-500 text-white font-semibold px-5 rounded-xl text-sm transition">Ask ↑</button>
    </form>
  </section>
</div>
<p class="text-center text-xs text-gray-600 pb-6">CampusNova answers only from the college dataset (RAG + LLM)</p>
<script>
  const box = document.getElementById('messages');
  const welcomeHTML = box.innerHTML;
  function scrollDown(){ box.scrollTop = box.scrollHeight; }
  function esc(s){ const d=document.createElement('div'); d.textContent=s; return d.innerHTML; }
  function addUser(){
    const q = document.getElementById('q').value;
    box.insertAdjacentHTML('beforeend',
      '<div class="flex justify-end"><div class="bg-indigo-600 text-white px-4 py-2 rounded-2xl rounded-tr-sm text-sm max-w-[80%]">'+esc(q)+'</div></div>');
    scrollDown();
  }
  function ask(q){
    document.getElementById('q').value = q;
    document.getElementById('chat-form').requestSubmit();
  }
  function clearChat(){ box.innerHTML = welcomeHTML; }
  document.body.addEventListener('htmx:afterSwap', scrollDown);
</script>
""".replace("__BUTTONS__", buttons).replace("__WELCOME__", welcome)
    return HTMLResponse(page("Chat", "chat", body))


@app.get("/dataset", response_class=HTMLResponse)
async def dataset_page():
    if DF.empty:
        rows = '<tr><td colspan="5" class="p-4 text-red-400">Dataset not loaded. Check data/campus_data.xlsx</td></tr>'
        count = 0
    else:
        rows = ""
        for i, r in DF.iterrows():
            rows += (
                '<tr class="border-t border-gray-800 hover:bg-gray-900">'
                f'<td class="p-3 text-gray-500">{i + 2}</td>'
                f'<td class="p-3"><span class="bg-indigo-950 text-indigo-300 text-xs px-2 py-1 rounded-full">{html.escape(str(r.get("Category", "")))}</span></td>'
                f'<td class="p-3 font-medium">{html.escape(str(r.get("Topic", "")))}</td>'
                f'<td class="p-3 text-gray-300">{html.escape(str(r.get("Details", "")))}</td>'
                f'<td class="p-3 text-gray-500">{html.escape(str(r.get("Department", "")))}</td></tr>'
            )
        count = len(DF)
    body = """
<div class="max-w-6xl mx-auto px-4 py-6">
  <div class="flex flex-col sm:flex-row sm:items-end justify-between gap-3 mb-4">
    <div>
      <h1 class="text-2xl font-bold">College Dataset</h1>
      <p class="text-sm text-gray-400">The Excel sheet CampusNova reads from — <strong>__COUNT__ rows</strong>. Each row becomes one searchable chunk.</p>
    </div>
    <input id="filter" placeholder="Search the dataset…" class="bg-gray-900 border border-gray-700 focus:border-indigo-500 outline-none rounded-xl px-4 py-2 text-sm sm:w-72">
  </div>
  <div class="overflow-x-auto border border-gray-800 rounded-2xl">
    <table class="w-full text-sm text-left" id="tbl">
      <thead class="bg-gray-900 text-gray-400 text-xs uppercase">
        <tr><th class="p-3">Row</th><th class="p-3">Category</th><th class="p-3">Topic</th><th class="p-3">Details</th><th class="p-3">Department</th></tr>
      </thead>
      <tbody>__ROWS__</tbody>
    </table>
  </div>
</div>
<script>
  document.getElementById('filter').addEventListener('input', e => {
    const v = e.target.value.toLowerCase();
    document.querySelectorAll('#tbl tbody tr').forEach(tr => {
      tr.style.display = tr.textContent.toLowerCase().includes(v) ? '' : 'none';
    });
  });
</script>
""".replace("__COUNT__", str(count)).replace("__ROWS__", rows)
    return HTMLResponse(page("Dataset", "dataset", body))


@app.get("/how-it-works", response_class=HTMLResponse)
async def how_page():
    def step(n, icon, title, text):
        return f"""
        <div class="bg-gray-900 border border-gray-800 rounded-2xl p-5">
          <div class="text-xs font-semibold text-indigo-400 mb-1">STEP {n}</div>
          <div class="text-2xl mb-2">{icon}</div>
          <h3 class="font-semibold mb-1">{title}</h3>
          <p class="text-sm text-gray-400">{text}</p>
        </div>"""
    steps = (
        step(1, "📊", "Dataset", "College information is stored in an Excel sheet: Category, Topic, Details, Department.")
        + step(2, "✂️", "Chunking", "Every row is turned into one text chunk, e.g. “[Row 19] Category: Hostel | Topic: Hostel fee | …”.")
        + step(3, "🔎", "Retrieval (RAG)", "The question is compared with every chunk using TF-IDF cosine similarity. The top 3 rows are selected.")
        + step(4, "🤖", "LLM answer", "The question and the 3 rows are sent to the LLM (OpenRouter). It must answer only from those rows.")
    )
    body = """
<div class="max-w-4xl mx-auto px-4 py-8">
  <h1 class="text-2xl font-bold mb-1">How CampusNova works</h1>
  <p class="text-gray-400 text-sm mb-6">RAG = Retrieval Augmented Generation. The model is not trained on the college data — it looks up the right rows first, then answers.</p>
  <div class="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-8">__STEPS__</div>

  <div class="bg-gray-900 border border-gray-800 rounded-2xl p-5 mb-6">
    <h2 class="font-semibold mb-3">Flow</h2>
    <div class="flex flex-wrap items-center gap-2 text-sm">
      <span class="bg-indigo-600 px-3 py-1 rounded-lg">Student question</span> →
      <span class="bg-gray-800 px-3 py-1 rounded-lg">Search Excel rows (RAG)</span> →
      <span class="bg-gray-800 px-3 py-1 rounded-lg">Top 3 rows</span> →
      <span class="bg-gray-800 px-3 py-1 rounded-lg">LLM</span> →
      <span class="bg-green-700 px-3 py-1 rounded-lg">Answer + sources</span>
    </div>
  </div>

  <div class="bg-gray-900 border border-gray-800 rounded-2xl p-5">
    <h2 class="font-semibold mb-3">Tech stack</h2>
    <div class="flex flex-wrap gap-2 text-xs">
      <span class="bg-gray-800 px-3 py-1 rounded-full">Python</span>
      <span class="bg-gray-800 px-3 py-1 rounded-full">FastAPI</span>
      <span class="bg-gray-800 px-3 py-1 rounded-full">HTMX</span>
      <span class="bg-gray-800 px-3 py-1 rounded-full">TailwindCSS</span>
      <span class="bg-gray-800 px-3 py-1 rounded-full">pandas + openpyxl</span>
      <span class="bg-gray-800 px-3 py-1 rounded-full">scikit-learn (TF-IDF)</span>
      <span class="bg-gray-800 px-3 py-1 rounded-full">OpenRouter LLM</span>
      <span class="bg-gray-800 px-3 py-1 rounded-full">Render</span>
    </div>
    <p class="text-sm text-gray-400 mt-4">Why RAG and not fine-tuning? Updating the answers only needs editing the Excel sheet — no retraining.</p>
  </div>
  <div class="text-center mt-8"><a href="/chat" class="bg-indigo-600 hover:bg-indigo-500 font-semibold px-6 py-3 rounded-xl transition inline-block">Try the chat →</a></div>
</div>
""".replace("__STEPS__", steps)
    return HTMLResponse(page("How it works", "how", body))


# ---------------------------------------------------------------- chat API
@app.post("/chat", response_class=HTMLResponse)
async def chat(request: Request, question: str = Form(...)):
    try:
        relevant_chunks = search_relevant_chunks(question, top_k=3)

        if not relevant_chunks:
            return HTMLResponse(bot_card(
                '<p class="text-sm text-yellow-300">I couldn\'t find this in the dataset. '
                'Please contact the college office.</p>',
                "Not in dataset", "text-yellow-300 bg-yellow-950"))

        context = "\n".join(c["text"] for c in relevant_chunks)

        system_prompt = f"""You are CampusNova, an intelligent and helpful AI assistant for college students.
Answer ONLY using the information in the context below.
If answer not found say: "I couldn't find this in the dataset. Please contact the college office."
Be friendly, professional and student-friendly.
Use bullet points and bold for important numbers.

Context from dataset:
{context}"""

        if not API_KEY:
            return HTMLResponse(bot_card(
                '<p class="text-sm text-red-300">OPENROUTER_API_KEY is not set on the server.</p>',
                "Config error", "text-red-300 bg-red-950"))

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "openrouter/auto",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
            },
            timeout=30,
        )
        data = response.json()
        if "choices" in data:
            answer = data["choices"][0]["message"]["content"]
        else:
            answer = "Sorry, I couldn't process your question. Please try again."

        inner = f"""
        <div class="answer text-sm text-gray-200 leading-relaxed">{to_html(answer)}</div>
        <div class="mt-3 flex flex-wrap items-center gap-2">
          <span class="text-xs text-gray-500">Retrieved rows:</span>
          {source_chips(relevant_chunks)}
        </div>"""
        return HTMLResponse(bot_card(inner))

    except Exception as e:
        return HTMLResponse(bot_card(
            f'<p class="text-sm text-red-300">Error: {html.escape(str(e))}</p>',
            "Error", "text-red-300 bg-red-950"))
