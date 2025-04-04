# 限定知識機器人

本專案是一套使用 **Python**、**Gradio** 與 **Gemini API** 打造的文件問答應用。  
透過向量檢索（RAG）技術，可上傳多份 PDF 文件，自動斷句、嵌入並建構知識庫，  
讓使用者針對文件內容進行自然語言提問，並由 Gemini 模型產出精準回應。

---

## 系統架構概覽

```text
使用者輸入問題
        ↓
 [語意向量化]
        ↓
與向量庫做 KNN 檢索（取 Top-K chunk）
        ↓
將相關段落作為 context 串進 Gemini prompt
        ↓
      Gemini 回答生成
