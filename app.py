import os
import re
import html
import json
import requests
import pandas as pd
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from rag import process_and_store_excel, search_relevant_chunks

app = FastAPI()
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
DATA_FILE = "data/campus_data.xlsx"
DF = pd.DataFrame()

ICONS = {
    "admission": "🎓", "fees": "💰", "fee": "💰", "scholarship": "🏅", "hostel": "🏠",
    "exams": "📝", "exam": "📝", "placement": "💼", "facilities": "🏛️", "contact": "📞",
}


def icon_for(cat: str) -> str:
    return ICONS.get(cat.strip().lower(), "📌")


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


# ---------------------------------------------------------------- PWA (installable app)
@app.get("/manifest.json")
async def manifest():
    return JSONResponse({
        "name": "CampusNova",
        "short_name": "CampusNova",
        "description": "AI College Assistant",
        "start_url": "/chat",
        "scope": "/",
        "display": "standalone",
        "background_color": "#05060f",
        "theme_color": "#05060f",
        "icons": [
            {"src": "/icon.svg", "sizes": "any", "type": "image/svg+xml", "purpose": "any"},
            {"src": "/icon.svg", "sizes": "512x512", "type": "image/svg+xml", "purpose": "maskable"},
        ],
    })


@app.get("/icon.svg")
async def icon():
    svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#6366f1"/><stop offset="1" stop-color="#a855f7"/></linearGradient></defs>
<rect width="512" height="512" fill="url(#g)"/>
<path d="M256 96l30 106 106-30-80 84 108 24-108 24 80 84-106-30-30 106-30-106-106 30 80-84-108-24 108-24-80-84 106 30z" fill="#fff" opacity=".95"/>
</svg>"""
    return Response(svg, media_type="image/svg+xml")


@app.get("/sw.js")
async def sw():
    js = "self.addEventListener('install',e=>self.skipWaiting());self.addEventListener('activate',e=>self.clients.claim());self.addEventListener('fetch',()=>{});"
    return Response(js, media_type="application/javascript")


HEAD_COMMON = """
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
  <meta name="theme-color" content="#05060f">
  <meta name="apple-mobile-web-app-capable" content="yes">
  <meta name="mobile-web-app-capable" content="yes">
  <link rel="manifest" href="/manifest.json">
  <link rel="icon" href="/icon.svg" type="image/svg+xml">
  <link rel="apple-touch-icon" href="/icon.svg">
  <script src="https://cdn.tailwindcss.com"></script>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
"""


# ---------------------------------------------------------------- simple layout (About / Dataset pages)
def page(title: str, active: str, body: str) -> str:
    def link(href, label, key):
        cls = "text-white bg-white/10" if key == active else "text-gray-400 hover:text-white"
        return f'<a href="{href}" class="px-3 py-2 rounded-lg text-sm font-medium transition {cls}">{label}</a>'

    nav = link("/dataset", "Dataset", "dataset") + link("/how-it-works", "How it works", "how")
    return """<!DOCTYPE html>
<html lang="en"><head>__HEAD__
  <title>__TITLE__ — CampusNova</title>
  <style>body{font-family:'Inter',sans-serif;background:#05060f}</style>
</head>
<body class="text-white min-h-screen flex flex-col">
  <header class="border-b border-white/10 sticky top-0 z-50 backdrop-blur bg-black/60">
    <div class="max-w-6xl mx-auto px-4 py-3 flex items-center justify-between gap-4">
      <a href="/chat" class="flex items-center gap-3">
        <div class="w-9 h-9 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-xl flex items-center justify-center font-bold text-sm">CN</div>
        <div class="font-bold">CampusNova</div>
      </a>
      <nav class="flex items-center gap-1">__NAV__
        <a href="/chat" class="ml-2 bg-indigo-600 hover:bg-indigo-500 px-4 py-2 rounded-lg text-sm font-semibold transition">← Back to chat</a>
      </nav>
    </div>
  </header>
  <main class="flex-1 w-full">__BODY__</main>
</body></html>""".replace("__HEAD__", HEAD_COMMON).replace("__TITLE__", title).replace("__NAV__", nav).replace("__BODY__", body)


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


def sources_block(chunks) -> str:
    chips, seen = "", set()
    for c in chunks:
        m = re.match(r"\[Row (\d+)\]\s*Category:\s*([^|]*)\|\s*Topic:\s*([^|]*)", c["text"])
        label = m.group(3).strip() if m else "College information"
        if label not in seen:
            seen.add(label)
            chips += ('<span class="bg-white/5 border border-white/10 text-gray-300 text-xs px-2.5 py-1 rounded-full">'
                      + html.escape(label) + "</span>")
    return (f'<div class="mt-3 pt-3 border-t border-white/10"><div class="text-[10px] font-semibold text-gray-500 uppercase tracking-widest mb-2">Sources</div>'
            f'<div class="flex flex-wrap gap-2">{chips}</div></div>')


def bot_msg(inner: str, actions: bool = False) -> str:
    footer = ""
    if actions:
        inner = ('<div class="flex items-center gap-2 mb-2 text-[10px] font-semibold tracking-wider">'
                 '<span class="px-2 py-0.5 rounded-full bg-indigo-500/10 border border-indigo-400/30 text-indigo-300">🔎 RETRIEVED (RAG)</span>'
                 '<span class="text-gray-600">→</span>'
                 '<span class="px-2 py-0.5 rounded-full bg-purple-500/10 border border-purple-400/30 text-purple-300">🤖 ANSWERED (LLM)</span></div>') + inner
        footer = ('<div class="mt-2 flex items-center justify-between text-[11px] text-gray-600">'
                  '<span data-ts></span>'
                  '<button type="button" onclick="copyAns(this)" class="hover:text-gray-300 transition">⧉ Copy answer</button></div>')
    return f"""
    <div class="msg flex items-start gap-3">
      <div class="w-8 h-8 shrink-0 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-xl flex items-center justify-center text-[11px] font-bold shadow-lg shadow-indigo-900/40">CN</div>
      <div class="bg-white/[0.04] border border-white/10 rounded-2xl rounded-tl-md px-4 py-3 max-w-[88%] text-[15px] leading-relaxed">
        {inner}{footer}
      </div>
    </div>"""


# ---------------------------------------------------------------- landing
@app.get("/", response_class=HTMLResponse)
async def home():
    with open("templates/landing.html", encoding="utf-8") as f:
        return HTMLResponse(f.read())


# ---------------------------------------------------------------- the app
@app.get("/chat", response_class=HTMLResponse)
async def chat_page():
    cats = list(dict.fromkeys(DF["Category"])) if (not DF.empty and "Category" in DF) else \
        ["Admission", "Fees", "Scholarship", "Hostel", "Exams", "Placement", "Facilities", "Contact"]
    side_topics = "".join(
        f'<button type="button" onclick="topic(this.dataset.t)" data-t="{html.escape(c)}" '
        f'class="w-full flex items-center gap-3 text-left text-sm text-gray-300 hover:text-white hover:bg-white/5 rounded-xl px-3 py-2.5 transition">'
        f'<span class="text-base">{icon_for(c)}</span><span>{html.escape(c)}</span></button>'
        for c in cats
    )
    cards = "".join(
        f'<button type="button" onclick="topic(this.dataset.t)" data-t="{html.escape(c)}" '
        f'class="group text-left bg-white/[0.04] hover:bg-white/[0.08] border border-white/10 hover:border-indigo-400/50 rounded-2xl p-4 transition">'
        f'<div class="text-2xl mb-2">{icon_for(c)}</div>'
        f'<div class="font-semibold text-sm">{html.escape(c)}</div>'
        f'<div class="text-xs text-gray-500 mt-0.5 group-hover:text-gray-400">Ask about {html.escape(c.lower())}</div></button>'
        for c in cats
    )
    rows = int(len(DF))

    return HTMLResponse("""<!DOCTYPE html>
<html lang="en"><head>__HEAD__
  <title>CampusNova — AI College Assistant</title>
  <script src="https://unpkg.com/htmx.org@1.9.12"></script>
  <style>
    html,body{height:100%}
    body{font-family:'Inter',sans-serif;background:#05060f;overflow:hidden}
    .glow{background:radial-gradient(60% 50% at 50% 0%,rgba(99,102,241,.25),transparent 70%),radial-gradient(40% 40% at 90% 100%,rgba(168,85,247,.15),transparent 70%)}
    .gradient-text{background:linear-gradient(135deg,#818cf8,#c084fc);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
    .answer ul{list-style:disc;padding-left:1.25rem;margin:.4rem 0}
    .answer p{margin:.35rem 0}
    .answer strong{color:#fff}
    .msg{animation:rise .25s ease-out}
    @keyframes rise{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
    .htmx-indicator{display:none}
    .htmx-request.htmx-indicator,.htmx-request .htmx-indicator{display:flex}
    .dot{width:6px;height:6px;border-radius:9999px;background:#a5b4fc;animation:b 1.2s infinite ease-in-out}
    .dot:nth-child(2){animation-delay:.15s}.dot:nth-child(3){animation-delay:.3s}
    @keyframes b{0%,80%,100%{opacity:.25;transform:scale(.8)}40%{opacity:1;transform:scale(1.15)}}
    #messages::-webkit-scrollbar{width:8px}#messages::-webkit-scrollbar-thumb{background:#1f2140;border-radius:8px}
  </style>
</head>
<body class="text-white">
<div class="h-full flex relative">

  <!-- Sidebar -->
  <div id="overlay" onclick="toggleMenu()" class="hidden fixed inset-0 bg-black/60 z-30 lg:hidden"></div>
  <aside id="sidebar" class="fixed lg:static inset-y-0 left-0 z-40 w-72 -translate-x-full lg:translate-x-0 transition-transform bg-[#080a18] border-r border-white/10 flex flex-col">
    <div class="p-4 flex items-center gap-3">
      <div class="w-10 h-10 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-xl flex items-center justify-center font-bold shadow-lg shadow-indigo-900/50">CN</div>
      <div class="leading-tight">
        <div class="font-bold text-lg">CampusNova</div>
        <div class="text-xs text-gray-500">AI College Assistant</div>
      </div>
    </div>
    <div class="px-4">
      <button type="button" onclick="newChat()" class="w-full bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-white font-semibold text-sm rounded-xl px-4 py-3 transition shadow-lg shadow-indigo-900/40">＋ New chat</button>
    </div>
    <div class="px-4 mt-6 mb-2 text-[11px] font-semibold text-gray-500 uppercase tracking-widest">Browse topics</div>
    <nav class="px-2 flex-1 overflow-y-auto">__SIDE__</nav>
    <div class="mx-3 mb-3 p-3 rounded-xl bg-indigo-500/10 border border-indigo-400/20">
      <div class="text-xs font-semibold text-indigo-200 mb-1">✦ Grounded answers</div>
      <p class="text-[11px] leading-relaxed text-gray-400">CampusNova first <b class="text-gray-300">searches</b> the college information (RAG), then an <b class="text-gray-300">LLM</b> writes the answer from only what it found — and shows the sources.</p>
    </div>
    <div class="p-4 border-t border-white/10 text-xs text-gray-500 space-y-2">
      <div class="flex items-center gap-2"><span class="w-2 h-2 rounded-full bg-green-400 animate-pulse"></span> Online · Available 24/7</div>
      <a href="/how-it-works" class="block hover:text-gray-300 transition">ⓘ About this project</a>
    </div>
  </aside>

  <!-- Main -->
  <main class="flex-1 min-w-0 flex flex-col glow">
    <header class="flex items-center gap-3 px-4 py-3 border-b border-white/10 backdrop-blur">
      <button type="button" onclick="toggleMenu()" class="lg:hidden w-9 h-9 rounded-lg hover:bg-white/10 flex items-center justify-center text-xl">☰</button>
      <div class="font-semibold">Assistant</div>
      <span class="hidden sm:flex items-center gap-1.5 text-[10px] font-semibold tracking-widest text-emerald-300 whitespace-nowrap"><span class="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>GROUNDED AI</span>
      <div class="ml-auto flex items-center gap-2 text-[10px] font-semibold tracking-wider whitespace-nowrap">
        <span class="px-2.5 py-1 rounded-full bg-indigo-500/10 border border-indigo-400/30 text-indigo-300">🔎 RAG</span>
        <span class="px-2.5 py-1 rounded-full bg-purple-500/10 border border-purple-400/30 text-purple-300">🤖 LLM</span>
      </div>
    </header>

    <div id="messages" class="flex-1 overflow-y-auto">
      <div class="max-w-3xl mx-auto px-4 py-6 flex flex-col gap-5" id="thread">

        <div id="hero" class="pt-6 sm:pt-14 pb-2 text-center">
          <div class="flex items-center justify-center gap-3 mb-5 text-[11px] font-semibold tracking-[0.25em] text-indigo-300">
            <span class="w-8 h-px bg-indigo-400/50"></span>POWERED BY RAG + LLM<span class="w-8 h-px bg-indigo-400/50"></span>
          </div>
          <div class="w-16 h-16 mx-auto mb-5 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-2xl flex items-center justify-center text-xl font-bold shadow-2xl shadow-indigo-900/60">CN</div>
          <h1 class="text-3xl sm:text-4xl font-extrabold tracking-tight"><span id="greet">Hello</span>,<br><span class="gradient-text">how can I help you today?</span></h1>
          <p class="text-gray-400 mt-3 text-sm sm:text-base">Ask me anything about your college — in your own words.</p>
          <div class="flex flex-wrap justify-center gap-x-5 gap-y-1 mt-4 text-xs text-gray-500">
            <span><span class="text-emerald-400">✓</span> Sources shown with every answer</span>
            <span><span class="text-emerald-400">✓</span> No guessing</span>
            <span><span class="text-emerald-400">✓</span> Available 24/7</span>
          </div>
          <div class="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-8 text-left">__CARDS__</div>
        </div>

        <div id="thinking" class="htmx-indicator msg items-start gap-3">
          <div class="w-8 h-8 shrink-0 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-xl flex items-center justify-center text-[11px] font-bold">CN</div>
          <div class="bg-white/[0.04] border border-white/10 rounded-2xl rounded-tl-md px-4 py-3">
            <div class="flex items-center gap-1.5"><span class="dot"></span><span class="dot"></span><span class="dot"></span></div>
            <div class="text-[11px] text-gray-500 mt-2">🔎 Searching college information → 🤖 writing your answer…</div>
          </div>
        </div>
      </div>
    </div>

    <div class="px-4 pb-4 pt-2" style="padding-bottom:max(1rem,env(safe-area-inset-bottom))">
      <form id="chat-form" hx-post="/chat" hx-target="#thinking" hx-swap="beforebegin" hx-indicator="#thinking" hx-disabled-elt="#send"
            hx-on::before-request="addUser()" hx-on::after-request="this.reset(); scrollDown()"
            class="max-w-3xl mx-auto flex items-center gap-2 bg-white/[0.06] border border-white/15 focus-within:border-indigo-400 rounded-2xl p-2 pl-4 shadow-xl shadow-black/40 transition">
        <input id="q" name="question" required autocomplete="off" placeholder="Ask about admissions, fees, hostel, exams, placements…"
               class="flex-1 min-w-0 bg-transparent outline-none text-[15px] placeholder-gray-500 py-2">
        <button id="send" class="shrink-0 w-11 h-11 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 hover:from-indigo-400 hover:to-purple-500 disabled:opacity-40 flex items-center justify-center text-lg transition">↑</button>
      </form>
      <p class="text-center text-[11px] text-gray-600 mt-2">CampusNova answers from the college knowledge base and can make mistakes. Please confirm important details with the college office.</p>
    </div>
  </main>
</div>

<script>
  const msgs = document.getElementById('messages');
  const thread = document.getElementById('thread');
  const heroHTML = document.getElementById('hero').outerHTML;
  const h = new Date().getHours();
  document.getElementById('greet').textContent = h < 12 ? 'Good morning' : (h < 17 ? 'Good afternoon' : 'Good evening');

  function scrollDown(){ msgs.scrollTo({top: msgs.scrollHeight, behavior: 'smooth'}); }
  function esc(s){ const d=document.createElement('div'); d.textContent=s; return d.innerHTML; }
  function hideHero(){ const x=document.getElementById('hero'); if(x) x.remove(); }
  function addUser(){
    hideHero();
    const q = document.getElementById('q').value;
    document.getElementById('thinking').insertAdjacentHTML('beforebegin',
      '<div class="msg flex justify-end"><div class="bg-gradient-to-br from-indigo-600 to-purple-600 text-white px-4 py-2.5 rounded-2xl rounded-tr-md text-[15px] max-w-[85%] shadow-lg shadow-indigo-900/30">'+esc(q)+'</div></div>');
    scrollDown();
  }
  function ask(q){ document.getElementById('q').value = q; document.getElementById('chat-form').requestSubmit(); closeMenu(); }
  function topic(t){ ask('Tell me about ' + t.toLowerCase()); }
  function newChat(){
    thread.querySelectorAll('.msg:not(#thinking)').forEach(e=>e.remove());
    if(!document.getElementById('hero')) thread.insertAdjacentHTML('afterbegin', heroHTML);
    closeMenu(); document.getElementById('q').focus();
  }
  const sb=document.getElementById('sidebar'), ov=document.getElementById('overlay');
  function toggleMenu(){ sb.classList.toggle('-translate-x-full'); ov.classList.toggle('hidden'); }
  function closeMenu(){ sb.classList.add('-translate-x-full'); ov.classList.add('hidden'); }
  function stamp(){
    const t = new Date().toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'});
    document.querySelectorAll('[data-ts]:empty').forEach(e => e.textContent = 'CampusNova · ' + t);
  }
  function copyAns(btn){
    const txt = btn.closest('.msg').querySelector('.answer').innerText;
    navigator.clipboard.writeText(txt).then(() => {
      const old = btn.textContent; btn.textContent = '✓ Copied';
      setTimeout(() => btn.textContent = old, 1500);
    }).catch(()=>{});
  }
  document.body.addEventListener('htmx:afterSwap', () => { stamp(); scrollDown(); });
  if('serviceWorker' in navigator){ navigator.serviceWorker.register('/sw.js').catch(()=>{}); }
</script>
</body></html>""".replace("__HEAD__", HEAD_COMMON).replace("__SIDE__", side_topics)
        .replace("__CARDS__", cards).replace("__ROWS__", str(rows)))


# ---------------------------------------------------------------- about pages (for the project demo)
@app.get("/dataset", response_class=HTMLResponse)
async def dataset_page():
    if DF.empty:
        rows = '<tr><td colspan="5" class="p-4 text-red-400">Dataset not loaded. Check data/campus_data.xlsx</td></tr>'
        count = 0
    else:
        rows = ""
        for i, r in DF.iterrows():
            rows += (
                '<tr class="border-t border-white/10 hover:bg-white/5">'
                f'<td class="p-3 text-gray-500">{i + 2}</td>'
                f'<td class="p-3"><span class="bg-indigo-500/10 text-indigo-300 text-xs px-2 py-1 rounded-full">{html.escape(str(r.get("Category", "")))}</span></td>'
                f'<td class="p-3 font-medium">{html.escape(str(r.get("Topic", "")))}</td>'
                f'<td class="p-3 text-gray-300">{html.escape(str(r.get("Details", "")))}</td>'
                f'<td class="p-3 text-gray-500">{html.escape(str(r.get("Department", "")))}</td></tr>'
            )
        count = len(DF)
    body = """
<div class="max-w-6xl mx-auto px-4 py-6">
  <div class="flex flex-col sm:flex-row sm:items-end justify-between gap-3 mb-4">
    <div>
      <h1 class="text-2xl font-bold">College Knowledge Base</h1>
      <p class="text-sm text-gray-400">The Excel dataset CampusNova reads from — <strong>__COUNT__ rows</strong>. Each row becomes one searchable chunk.</p>
    </div>
    <input id="filter" placeholder="Search the dataset…" class="bg-white/5 border border-white/15 focus:border-indigo-400 outline-none rounded-xl px-4 py-2 text-sm sm:w-72">
  </div>
  <div class="overflow-x-auto border border-white/10 rounded-2xl">
    <table class="w-full text-sm text-left" id="tbl">
      <thead class="bg-white/5 text-gray-400 text-xs uppercase">
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
        <div class="bg-white/[0.04] border border-white/10 rounded-2xl p-5">
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

  <div class="bg-white/[0.04] border border-white/10 rounded-2xl p-5 mb-6">
    <h2 class="font-semibold mb-3">Flow</h2>
    <div class="flex flex-wrap items-center gap-2 text-sm">
      <span class="bg-indigo-600 px-3 py-1 rounded-lg">Student question</span> →
      <span class="bg-white/10 px-3 py-1 rounded-lg">Search Excel rows (RAG)</span> →
      <span class="bg-white/10 px-3 py-1 rounded-lg">Top 3 rows</span> →
      <span class="bg-white/10 px-3 py-1 rounded-lg">LLM</span> →
      <span class="bg-green-700 px-3 py-1 rounded-lg">Answer + sources</span>
    </div>
  </div>

  <div class="bg-white/[0.04] border border-white/10 rounded-2xl p-5">
    <h2 class="font-semibold mb-3">Tech stack</h2>
    <div class="flex flex-wrap gap-2 text-xs">
      <span class="bg-white/10 px-3 py-1 rounded-full">Python</span>
      <span class="bg-white/10 px-3 py-1 rounded-full">FastAPI</span>
      <span class="bg-white/10 px-3 py-1 rounded-full">HTMX</span>
      <span class="bg-white/10 px-3 py-1 rounded-full">TailwindCSS</span>
      <span class="bg-white/10 px-3 py-1 rounded-full">pandas + openpyxl</span>
      <span class="bg-white/10 px-3 py-1 rounded-full">scikit-learn (TF-IDF)</span>
      <span class="bg-white/10 px-3 py-1 rounded-full">OpenRouter LLM</span>
      <span class="bg-white/10 px-3 py-1 rounded-full">Installable web app (PWA)</span>
      <span class="bg-white/10 px-3 py-1 rounded-full">Render</span>
    </div>
    <p class="text-sm text-gray-400 mt-4">Why RAG and not fine-tuning? Updating the answers only needs editing the Excel sheet — no retraining.</p>
  </div>
</div>
""".replace("__STEPS__", steps)
    return HTMLResponse(page("How it works", "how", body))


# ---------------------------------------------------------------- chat API
@app.post("/chat", response_class=HTMLResponse)
async def chat(request: Request, question: str = Form(...)):
    try:
        relevant_chunks = search_relevant_chunks(question, top_k=3)

        if not relevant_chunks:
            return HTMLResponse(bot_msg(
                '<p class="text-yellow-200">I couldn\'t find this in my knowledge base. '
                'Please contact the college office for help.</p>'))

        context = "\n".join(c["text"] for c in relevant_chunks)

        system_prompt = f"""You are CampusNova, an intelligent and helpful AI assistant for college students.
Answer ONLY using the information in the context below.
If answer not found say: "I couldn't find this in the dataset. Please contact the college office."
Be friendly, professional and student-friendly.
Use bullet points and bold for important numbers.

Context from dataset:
{context}"""

        if not API_KEY:
            return HTMLResponse(bot_msg(
                '<p class="text-red-300">OPENROUTER_API_KEY is not set on the server.</p>'))

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

        inner = f'<div class="answer text-gray-100">{to_html(answer)}</div>{sources_block(relevant_chunks)}'
        return HTMLResponse(bot_msg(inner, actions=True))

    except Exception as e:
        return HTMLResponse(bot_msg(f'<p class="text-red-300">Something went wrong: {html.escape(str(e))}</p>'))
