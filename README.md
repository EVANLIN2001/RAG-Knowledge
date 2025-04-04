# 限定知識機器人

本專案是一套使用 **Python**、**Gradio** 與 **Gemini API** 打造的文件問答應用。透過向量檢索（RAG）技術，可上傳多份 PDF 文件，自動斷句、嵌入並建構知識庫，讓使用者針對文件內容進行自然語言提問，並由 Gemini 模型產出精準回應。

## 核心功能說明

### 多 PDF 上傳與斷句切分
- 使用 `PyMuPDF` 擷取頁面內容  
- 搭配 `nltk` 將文字切成固定長度段落（chunk）  
- 每段記錄包含：檔名、頁碼、文字內容  

### 向量建構與檢索
- 使用 `sentence-transformers` 對每段文字進行語意嵌入  
- 建立 `sklearn.NearestNeighbors` 向量索引  
- 根據提問找出最相關的 Top-K 內容段落  

### Gemini 模型整合
- 串接 Gemini 2.0 Flash（via `google.generativeai`）  
- 將檢索段落 + 問題組成 prompt 傳入  
- 回應格式精簡、口吻專業，強化可讀性  

### Gradio UI + 深色主題介面
- 支援多 PDF 上傳與即時問答  
- 可調整 K 值影響回覆精準度  
- 深色主題設計，提升視覺與可讀體驗  
