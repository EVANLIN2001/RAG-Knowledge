# 限定知識機器人

本專案使用 **Python**、**Gradio** 與 **Gemini** 打造的文件問答應用。透過向量檢索（RAG）技術，可上傳多份 PDF 文件，自動斷句、嵌入並建構知識庫，讓使用者針對文件內容進行自然語言提問，並由 Gemini 模型產出精準回應。

<img src="https://github.com/EVANLIN2001/RAG-Knowledge/blob/main/image/%E6%88%AA%E5%9C%96%202025-04-04%20%E4%B8%8B%E5%8D%885.40.53.png" alt="Demo 1" width="800"><br>

<img src="https://github.com/EVANLIN2001/RAG-Knowledge/blob/main/image/%E6%88%AA%E5%9C%96%202025-04-04%20%E4%B8%8B%E5%8D%885.41.24.png" alt="Demo 1" width="800"><br>

Uploading 限定知識機器人_Demo影片.mov…

## 使用流程概覽

```text
使用者上傳 PDF 文件
        ↓
  PDF 每頁擷取文字（PyMuPDF）
        ↓
  文字斷句切分成 chunk（nltk）
        ↓
  每段內容進行語意嵌入（MiniLM-L6-v2）
        ↓
  建立向量索引（NearestNeighbors）
        ↓
使用者輸入問題 → 轉換為語意向量
        ↓
  檢索最相關的 Top-K chunk
        ↓
  將段落作為 context 串進 Gemini prompt
        ↓
          Gemini 回答生成
```

## 核心功能說明

### 多 PDF 上傳與斷句切分
- 使用 `PyMuPDF` 擷取頁面內容  
- 搭配 `nltk` 將文字切成固定長度段落（chunk）  

### 向量建構與檢索
- 使用 `sentence-transformers` 對每段文字進行語意嵌入  
- 建立 `sklearn.NearestNeighbors` 向量索引  
- 根據提問找出最相關的 Top-K 內容段落  

### Gemini 模型整合
- 串接 Gemini 2.0 Flash
- 將檢索段落 + 問題組成 prompt 傳入  
- 回應格式精簡、口吻專業，強化可讀性  

### Gradio UI
- 支援多 PDF 上傳與即時問答  
- 可調整 K 值影響回覆精準度

## 技術棧
- **語言**：Python  
- **PDF 處理**：PyMuPDF  
- **文本切分**：nltk  
- **語意嵌入**：sentence-transformers（MiniLM-L6-v2）  
- **向量檢索**：scikit-learn NearestNeighbors  
- **大語言模型**：Gemini 2.0 Flash
- **使用者介面**：Gradio

## 本專案結合 PDF 文件處理、語意嵌入、向量檢索與 Gemini 模型應用，架構清晰、模組化程度高，適合作為 RAG 技術在文件問答場景中的實作入門範例。
