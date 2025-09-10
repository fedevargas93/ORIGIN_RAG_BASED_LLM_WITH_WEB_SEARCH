!pip install gradio google-generativeai sentence-transformers faiss-cpu requests beautifulsoup4 numpy
!pip install google-search-results

import os
from google.colab import userdata

os.environ['GOOGLE_API_KEY'] = userdata.get('GOOGLE_API_KEY')

# Final, working agent.py code with SerpApi Integration
import argparse, io, json, os, re, sys, math, time, random
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from collections import deque
import numpy as np
import gradio as gr
import google.generativeai as genai

# -------------------- Utils --------------------
random.seed(7); np.random.seed(7)

def now_ts() -> float:
    return time.time()

def require(module: str, pip_name: Optional[str] = None):
    try: return __import__(module)
    except Exception as e:
        pname = pip_name or module
        raise RuntimeError(f"Missing dependency '{module}'. Install: pip install {pname}") from e

def _strip_boilerplate(text: str) -> str:
    text = re.sub(r'(?is)', '', text)
    boilerplate_patterns = [
        r'Subscribe To Newsletters', r'View All Billionaires', r'Forbes 400',
        r'Money & Politics', r'Innovation', r'Leadership', r'Billionaires',
        r'all rights reserved', r'© \d{4}', r'site map', r'skip to main content',
        r'log in', r'sign in', r'sign up', r'careers', r'about us', r'contact us'
    ]
    for pat in boilerplate_patterns:
        text = re.sub(pat, ' ', text, flags=re.IGNORECASE)
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if not line or len(line.split()) < 4: continue
        cleaned_lines.append(line)
    text = " ".join(cleaned_lines)
    return re.sub(r'\s+', ' ', text).strip()

# -------------------- Classical Stack & Web Fetching --------------------
class ClassicalStack:
    def __init__(self, emb_model="sentence-transformers/all-MiniLM-L6-v2"):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set. Please set it in Colab Secrets.")
        genai.configure(api_key=api_key)
        self.gen_model = genai.GenerativeModel('gemini-1.5-flash')
        st=require("sentence_transformers","sentence-transformers"); self.emb=st.SentenceTransformer(emb_model)
        faiss=require("faiss","faiss-cpu"); self.faiss=faiss; self.index=None
        self.corpus:List[str]=[]; self.meta:List[Dict[str,Any]]=[]

    def add_texts(self, texts: List[str], metas: Optional[List[Dict]] = None):
        if not texts: return
        if metas is None: metas=[{}]*len(texts)
        vecs=self.emb.encode(texts, normalize_embeddings=True).astype("float32")
        if self.index is None: self.index=self.faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs); self.corpus.extend(texts); self.meta.extend(metas)

    def search(self, query: str, k: int = 8) -> List[tuple]:
        if self.index is None: return []
        qv=self.emb.encode([query], normalize_embeddings=True).astype("float32")
        D,I=self.index.search(qv, k); out=[]
        for d,i in zip(D[0],I[0]):
            if int(i)!=-1: out.append((self.corpus[int(i)], float(d), int(i)))
        return out

    def generate(self, prompt: str) -> str:
        try:
            response = self.gen_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error communicating with the Generation API: {e}"

class WebIngestor:
    def __init__(self, classical_stack: ClassicalStack):
        self.classical=classical_stack; self.requests=require("requests"); self.bs4=require("bs4", "beautifulsoup4")
    def _chunk(self, t: str, size: int = 1200, overlap: int = 150) -> List[str]:
        t=t.strip(); i=0; out=[]
        while i<len(t): out.append(t[i:i+size]); i+=size-overlap
        return out
    def fetch_and_ingest(self, url: str) -> str:
        try:
            headers={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            r=self.requests.get(url, timeout=20, headers=headers); r.raise_for_status()
            soup=self.bs4.BeautifulSoup(r.content, "html.parser", from_encoding="utf-8")
            for s in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]): s.decompose()
            text=soup.get_text(" ", strip=True)
            text=_strip_boilerplate(text)
            if not text or len(text) < 100: return f"Fetch OK, but no meaningful content after cleaning: {url}"
            ch=self._chunk(text); metas=[{"source":"web","url":url, "ts":now_ts()} for _ in ch]
            self.classical.add_texts(ch, metas)
            return f"Fetched+ingested {len(ch)} chunks from {url}"
        except Exception as e: return f"Fetch failed: {e}"

# --- MODIFIED: Upgraded WebSearch with SerpApi ---
class WebSearch:
    def __init__(self, web_ingestor: WebIngestor):
        self.web=web_ingestor
        self.requests=require("requests")
        self.serpapi_client = require("serpapi")
        self.api_key = None

    def set_api_key(self, api_key: Optional[str]):
        self.api_key = api_key

    def _serpapi_search(self, query: str, max_results: int) -> List[str]:
        print(f"--- Performing search with SerpApi for query: {query} ---")
        params = {
            "engine": "google",
            "q": query,
            "api_key": self.api_key
        }
        search = self.serpapi_client.GoogleSearch(params)
        results = search.get_dict()
        organic_results = results.get("organic_results", [])
        return [res.get("link") for res in organic_results[:max_results] if res.get("link")]

    def _ddg_search(self, query: str, max_results: int) -> List[str]:
        print(f"--- Performing search with DuckDuckGo for query: {query} ---")
        try:
            headers={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            resp=self.requests.get("https://html.duckduckgo.com/html/", params={"q":query}, timeout=15, headers=headers)
            resp.raise_for_status()
            urls=re.findall(r'a class="result__a" href="([^"]+)"', resp.text)
            return list(dict.fromkeys(urls))[:max_results]
        except Exception: return []

    def search_urls(self, query: str, max_results: int = 4) -> List[str]:
        # Prioritize SerpApi if the key is available
        if self.api_key:
            try:
                urls = self._serpapi_search(query, max_results)
                if urls: return urls
            except Exception as e:
                print(f"SerpApi search failed: {e}. Falling back to DuckDuckGo.")

        # Fallback to DuckDuckGo
        return self._ddg_search(query, max_results)


# -------------------- Cognitive Core --------------------
@dataclass
class Episode: text:str; context:Dict[str,Any]; ts:float
class Hippocampus:
    def __init__(self, capacity=1000): self.episodes=deque(maxlen=capacity)
    def store(self, text: str, context: Dict): self.episodes.append(Episode(text, context, now_ts()))
    def last_n(self, n=10) -> List[Episode]: return list(self.episodes)[-n:]

class Controller:
    def route(self, text: str) -> Dict[str, Any]:
        low=text.lower().strip()
        if low.startswith("remember:"): return {"action":"store", "payload": text.split(":",1)[1]}
        if low.startswith("search:"): return {"action":"search", "payload": text.split(":",1)[1]}
        if low.startswith("origin:"): return {"action":"origin", "payload": text.split(":",1)[1]}
        if low.startswith("summarize:"): return {"action":"summarize", "payload": text.split(":",1)[1]}
        return {"action":"answer", "payload": text}

# -------------------- Agent --------------------
class HybridAgent:
    def __init__(self):
        self.classical=ClassicalStack()
        self.hipp=Hippocampus()
        self.web_ingestor=WebIngestor(self.classical)
        self.web_search=WebSearch(self.web_ingestor)
        self.ctrl=Controller()
        self.ingest_cursor=0
        self.hipp.store("I am a helpful AI assistant.", {"source":"seed"})
        self.classical.add_texts(["I am a helpful AI assistant."], [{"source":"seed"}])

    def set_serpapi_key(self, api_key: Optional[str]):
        """Allows the UI to configure the search API key."""
        self.web_search.set_api_key(api_key)

    def mark_ingest_start(self): self.ingest_cursor=len(self.classical.corpus)
    def get_last_ingested(self) -> tuple[List[str], List[Dict]]: return self.classical.corpus[self.ingest_cursor:], self.classical.meta[self.ingest_cursor:]

    def summarize_material(self, query: str, texts: List[str], metas: List[Dict]) -> str:
        if not texts: return "Error: No text was provided to summarize."
        sources = list(dict.fromkeys([m.get('url') for m in metas if m.get('url')]))
        material=" ".join(texts)[:8000]
        src_map="\n".join([f"[S{i+1}] {u}" for i, u in enumerate(sources)])
        prompt=(
            "You are a skilled synthesizer. Your task is to write a clean, human-readable summary based ONLY on the provided material. "
            "DO NOT copy and paste sentences. You MUST rewrite the information in your own words. "
            "Synthesize the key points into a new, original summary using the requested structure. Cite claims with [S#].\n\n"
            f"TOPIC: {query}\n\n"
            "REQUIRED FORMAT (markdown):\n"
            "## TL;DR\n- (A 2-3 sentence overview.)\n\n"
            "## Key Findings\n- (3-5 bullet points with the most important facts. Append [S#] to each.)\n\n"
            "## Sources\n" + src_map + "\n\n"
            "MATERIAL TO SYNTHESIZE:\n" + material
        )
        summary=self.classical.generate(prompt)
        self.hipp.store(summary, {"source":"summary", "query":query, "urls":sources})
        return summary

    def answer(self, prompt: str) -> Dict[str, str]:
        route=self.ctrl.route(prompt)
        action, payload=route["action"], route.get("payload", "").strip()
        if not payload: return {"text": "Please provide a command or question."}
        if action == "store":
            self.hipp.store(payload, {"source": "user_memory"}); self.classical.add_texts([payload], [{"source":"user_memory"}])
            return {"text": "Got it. I'll remember that."}
        if action == "search":
            self.mark_ingest_start(); urls=self.web_search.search_urls(payload)
            ingest_log="\n".join([self.web_ingestor.fetch_and_ingest(u) for u in urls])
            return {"text": f"[Learned from Search]\n{ingest_log}"}
        if action == "origin":
            self.mark_ingest_start(); urls=self.web_search.search_urls(payload)
            if not urls: return {"text": "My search returned no URLs."}
            ingest_log="\n".join([self.web_ingestor.fetch_and_ingest(u) for u in urls])
            new_texts, new_metas=self.get_last_ingested()
            if not new_texts: return {"text": f"Finished learning, but no usable content was found.\n\n[Ingestion Log]\n{ingest_log}"}
            summary=self.summarize_material(payload, new_texts, new_metas)
            return {"text": f"{summary}\n\n[Ingestion Log]\n{ingest_log}"}
        if action == "summarize":
            new_texts, new_metas=self.get_last_ingested()
            if not new_texts: return {"text": "There is nothing new to summarize."}
            summary=self.summarize_material(payload, new_texts, new_metas)
            return {"text": summary}
        if action == "answer":
            chunks=self.classical.search(payload, k=5)
            if not chunks: return {"text": "I don't have enough information to answer that."}
            context="\n\n".join([f"- {c[0]}" for c in chunks])
            prompt=f"Answer the user's QUESTION based ONLY on the following CONTEXT.\n\nCONTEXT:\n{context}\n\nQUESTION:\n{payload}\n\nANSWER:"
            return {"text": self.classical.generate(prompt)}
        return {"text": "Unknown command."}

# -------------------- UI --------------------
def run_ui(agent: HybridAgent):
    def handle_submit(message: str, history: List[tuple], serp_api_key: str) -> tuple[str, List[tuple]]:
        # Configure the agent with the API key from the UI on every request
        agent.set_serpapi_key(serp_api_key)

        if not message.strip(): return "", history
        response=agent.answer(message).get("text", "An error occurred.")
        history.append((message, response))
        return "", history

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Origin (Powered by Gemini & SerpApi)")

        with gr.Row():
            # --- MODIFIED: Added SerpApi Key Textbox to the UI ---
            serpapi_key_box = gr.Textbox(
                label="SerpApi Key (Optional)",
                placeholder="Enter your SerpApi key for premium search...",
                type="password"
            )

        chatbot=gr.Chatbot(height=500, label="Chat")
        msg_box=gr.Textbox(placeholder="Enter a command... (e.g., origin: history of Costa Rica)", label="Command")

        msg_box.submit(handle_submit, [msg_box, chatbot, serpapi_key_box], [msg_box, chatbot])

        with gr.Accordion("Commands & Debug", open=False):
            gr.Markdown("**Commands:**\n- `origin: [topic]`\n- `search: [topic]`\n- `summarize: [topic]`\n- `remember: [fact]`")
            debug_output=gr.Textbox(label="Last 10 Memory Episodes", lines=10, interactive=False)
            btn_dump=gr.Button("Dump Hippocampus")
            btn_dump.click(lambda: "\n".join([f"[{e.ts:.0f}] ({e.context.get('source','?')}) {e.text[:100]}..." for e in agent.hipp.last_n(10)]), outputs=[debug_output])

    demo.launch(share=True, debug=True)

# -------------------- Main (for Colab) --------------------
agent = HybridAgent()
run_ui(agent)
