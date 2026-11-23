import os
import requests
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from rag import process_and_store_excel, search_relevant_chunks

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

API_KEY = os.environ.get("OPENROUTER_API_KEY", "")


@app.on_event("startup")
async def load_dataset():
    try:
        chunks_path = os.path.join("vector_store", "chunks.pkl")
        if os.path.exists(chunks_path):
            os.remove(chunks_path)
        n = process_and_store_excel("data/campus_data.xlsx", "Campus Dataset", "General")
        print(f"Loaded {n} rows from dataset")
    except Exception as e:
        print("Dataset load failed:", e)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    with open("templates/landing.html", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    with open("templates/index.html", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.post("/chat", response_class=HTMLResponse)
async def chat(request: Request, question: str = Form(...)):
    try:
        relevant_chunks = search_relevant_chunks(question, top_k=3)

        if not relevant_chunks:
            return HTMLResponse("""
                <div class="bg-gray-800 rounded-xl p-4">
                    <p class="text-yellow-400">⚠️ I couldn't find this in the dataset.</p>
                </div>
            """)

        context = ""
        sources = []
        for chunk in relevant_chunks:
            context += f"\n{chunk['text']}\n"
            source = f"{chunk['doc_name']} ({chunk['category']})"
            if source not in sources:
                sources.append(source)

        system_prompt = f"""You are CampusNova, an intelligent and helpful AI assistant for college students.
Answer ONLY using the information in the context below.
If answer not found say: "I couldn't find this in the dataset. Please contact the college office."
Be friendly, professional and student-friendly.
Use bullet points and bold for important numbers.

Context from dataset:
{context}"""

        response = requests.post(
            'https://openrouter.ai/api/v1/chat/completions',
            headers={
                'Authorization': f'Bearer {API_KEY}',
                'Content-Type': 'application/json'
            },
            json={
                'model': 'openrouter/auto',
                'messages': [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': question}
                ]
            },
            timeout=30
        )

        data = response.json()
        if 'choices' in data:
            answer = data['choices'][0]['message']['content']
        else:
            answer = "Sorry, I couldn't process your question. Please try again."

        sources_html = ""
        for source in sources:
            sources_html += f'<span class="bg-teal-900 text-teal-300 text-xs px-2 py-1 rounded-full">{source}</span>'

        return HTMLResponse(f"""
            <div class="bg-gray-800 rounded-xl p-5 border border-gray-700">
                <div class="flex items-center gap-2 mb-3">
                    <div class="w-7 h-7 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-full flex items-center justify-center">
                        <span class="text-white text-xs">CN</span>
                    </div>
                    <span class="text-white font-semibold">CampusNova</span>
                    <span class="text-xs text-green-400 bg-green-900 px-2 py-0.5 rounded-full">From dataset</span>
                </div>
                <div class="text-gray-200 text-sm leading-relaxed">{answer}</div>
                <div class="mt-3 flex flex-wrap gap-2">
                    <span class="text-xs text-gray-500">Sources:</span>
                    {sources_html}
                </div>
            </div>
        """)

    except Exception as e:
        return HTMLResponse(f'<p class="text-red-400">Error: {str(e)}</p>')
