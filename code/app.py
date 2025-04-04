import os
# 設定環境變數以避免 macOS fork 安全性檢查引起的問題
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

import gradio as gr
import google.generativeai as genai
import fitz  # PyMuPDF
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import traceback
import nltk

# 下載 nltk 所需資源（若尚未下載）
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# 載入 .env 檔案中的環境變數
load_dotenv()

# 初始化 Gemini 模型
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
llm = genai.GenerativeModel("gemini-2.0-flash")

# 初始化 embedding 模型
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

class DocumentQA:
    def __init__(self, max_chars=500):
        self.neigh = None
        self.stored_chunks = []  # 每個 chunk 為 dict：{'filename': 檔案名稱, 'page': 頁碼, 'text': 內容}
        self.vectors = None
        self.max_chars = max_chars  # 每個 chunk 最大字數，可依需求調整

    def extract_text_from_pdf(self, file_path):
        """
        讀取 PDF，回傳列表，每個元素為 (filename, page_number, text)
        """
        doc = fitz.open(file_path)
        pages = []
        filename = os.path.basename(file_path)
        for i, page in enumerate(doc):
            text = page.get_text()
            pages.append((filename, i + 1, text))
        doc.close()
        return pages

    def chunk_text(self, pages):
        """
        利用 nltk 斷句，將每一頁文字拆分成多個 chunk，
        每個 chunk 限制最大字數，並附上來源檔名與頁碼
        """
        chunks = []
        for filename, page_num, text in pages:
            sentences = nltk.sent_tokenize(text)
            current_chunk = ""
            for sent in sentences:
                if len(current_chunk) + len(sent) < self.max_chars:
                    current_chunk += " " + sent
                else:
                    if current_chunk.strip():
                        chunks.append({"filename": filename, "page": page_num, "text": current_chunk.strip()})
                    current_chunk = sent
            if current_chunk.strip():
                chunks.append({"filename": filename, "page": page_num, "text": current_chunk.strip()})
        return chunks

    def embed_chunks(self, chunks):
        texts = [chunk["text"] for chunk in chunks]
        embeddings = embedding_model.encode(texts)
        return np.array(embeddings)

    def process_pdf(self, files):
        """
        多檔案處理：傳入的 files 為檔案路徑的列表
        """
        try:
            all_chunks = []
            for file_path in files:
                pages = self.extract_text_from_pdf(file_path)
                chunks = self.chunk_text(pages)
                all_chunks.extend(chunks)
            vectors = self.embed_chunks(all_chunks)
            self.neigh = NearestNeighbors(metric='euclidean')
            self.neigh.fit(vectors)
            self.stored_chunks = all_chunks
            self.vectors = vectors
            return f"✅ 已處理 {len(all_chunks)} 個 chunk，向量庫建立完成！"
        except Exception:
            return "⚠️ 錯誤：未偵測到上傳的 PDF 檔案，請重新上傳。"

    def chat_with_doc(self, query, k):
        """
        依據使用者輸入的問題與 k 值檢索相關 chunk，並呼叫 Gemini 模型生成回答
        """
        try:
            if not self.stored_chunks or self.neigh is None:
                return "⚠️ 請先上傳並處理 PDF 檔案！"
            k = int(k)
            k = min(k, len(self.stored_chunks))
            query_vec = embedding_model.encode([query])
            distances, indices = self.neigh.kneighbors(query_vec, n_neighbors=k)
            related_chunks = [self.stored_chunks[idx] for idx in indices[0] if 0 <= idx < len(self.stored_chunks)]
            if not related_chunks:
                return "⚠️ 查無相關段落，請換個問題試試。"
            context_lines = [f"【{chunk['filename']} 第 {chunk['page']} 頁】{chunk['text']}" for chunk in related_chunks]
            context = "\n\n".join(context_lines)
            prompt = (
                "你是一位專業且知識豐富的助理，根據以下文件內容摘要來回答使用者的問題：\n\n"
                f"{context}\n\n"
                f"使用者問題：{query}\n\n"
                "請提供一個簡潔且精確的回答。"
            )
            print("Debug: 發送給 Gemini 的 prompt 為：\n", prompt)
            response = llm.generate_content(prompt)
            print("Debug: Gemini 回傳結果：", response)
            if not response or not response.text:
                return "⚠️ 沒有得到回答，請稍後重試。"
            return response.text.strip()
        except Exception:
            return "⚠️ 錯誤：未偵測到上傳的 PDF 檔案，請重新上傳。"

doc_qa = DocumentQA(max_chars=500)

# 全新深色主題 CSS（高對比度、加粗字體）
custom_css = """
body { 
    background-color: #1e1e1e !important;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
    color: #ffffff !important;
    margin: 0;
    padding: 0;
}
.gradio-container {
    border-radius: 10px !important;
    box-shadow: 0 10px 40px rgba(0,0,0,0.8) !important;
    padding: 20px !important;
    background-color: #252526 !important;
    color: #ffffff !important;
}
header {
    background-color: #007acc !important;
    color: #ffffff !important;
    padding: 20px !important;
    border-radius: 10px 10px 0 0 !important;
    text-align: center !important;
}
header h1 {
    font-size: 48px !important;
    margin: 0 !important;
    font-weight: bold !important;
}
header p {
    font-size: 24px !important;
    margin: 10px 0 0 0 !important;
}
.tab-title {
    font-size: 24px !important;
    font-weight: bold !important;
    margin-bottom: 10px !important;
    color: #ffffff !important;
}
label, .gr-label {
    color: #ffffff !important;
    font-weight: bold !important;
}
.gr-button {
    background-color: #007acc !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 5px !important;
    padding: 10px 20px !important;
}
.gr-button:hover {
    background-color: #005f99 !important;
}
"""

with gr.Blocks(css=custom_css) as demo:
    gr.HTML("""
        <header>
            <h1>限定知識機器人</h1>
            <p>上傳 PDF，系統將透過 RAG 技術整合多份文件，並透過 Gemini 模型生成回應</p>
        </header>
    """)
    
    with gr.Tabs():
        with gr.TabItem("建立知識庫"):
            gr.Markdown("<div class='tab-title'>上傳 PDF 文件（支援複數檔案）</div>")
            with gr.Row():
                file_input = gr.File(label="選擇 PDF 檔案", file_count="multiple", type="filepath")
                process_status = gr.Textbox(label="處理狀態", lines=2, interactive=False)
            file_input.change(fn=doc_qa.process_pdf, inputs=file_input, outputs=process_status)
            gr.Markdown("上傳後系統將解析文件並建立向量索引，請稍候...")
            
        with gr.TabItem("限定問答"):
            gr.Markdown("<div class='tab-title'>請輸入您的問題，選擇 K 值後點擊【送出問題】</div>")
            with gr.Row():
                query_input = gr.Textbox(
                    label="您的問題", 
                    placeholder="例如：說明第 3 頁的內容", 
                    lines=2
                )
            k_slider = gr.Slider(minimum=1, maximum=10, step=1, value=3, label="選擇檢索數量（Top-K）")
            submit_button = gr.Button("送出問題")
            answer_output = gr.Textbox(label="AI 回答", lines=8)
            submit_button.click(fn=doc_qa.chat_with_doc, inputs=[query_input, k_slider], outputs=answer_output)

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    demo.launch(share=True)