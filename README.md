# Medical Image Analysis Agent

## Project Overview

This project builds a powerful **ReAct-based AI Agent** capable of analyzing **medical images**, **patient metadata**, and providing **long-term memory support**. The system uses **Google Gemini 2.0 Flash** for multimodal reasoning and **LangChain** to implement tool-using agents. Reports and patient conversations are stored in a **SQLite database** for personalized diagnostics and history-aware responses. The application is deployed using **Streamlit** for an interactive web UI.

---

## Introduction

Medical imaging is one of the most data-intensive and diagnostic-critical areas in healthcare. This system leverages the latest in **multimodal large language models (LLMs)** and **agentic architectures** to:

- Analyze medical images (e.g., chest X-rays, scans).
- Correlate them with patient symptoms and metadata.
- Generate structured, diagnostic reports.
- Store user-specific records for long-term access.
- Provide a conversational assistant to interact with medical history and reports.

---

## How It Works

### 1. Medical Image Analysis

- Users upload images in `PNG`, `JPG`, or `JPEG` formats.
- Gemini 2.0 Flash model processes the image **alongside user symptoms**.
- Extracted insights include:
  - Abnormalities and visible issues.
  - Probable medical conditions.
  - Affected organs.
  - Recommendations and prevention tips.

---

### 2. Medical Report Generation

- The insights from the image are **merged with patient metadata**:
  - Name, Age, Gender
  - Reported symptoms
- A **LangChain LLMChain** is used to generate a **structured medical report**.
- The report includes:
  - Title
  - Key findings
  - Diagnosis summary
  - Prevention/treatment recommendations

---

### 3. User-Specific Long-Term Memory

- Each patient interaction (metadata, image analysis, report, queries) is stored in a **SQLite database**.
- This serves as **persistent memory** for each user.
- Enables:
  - Retrieval of previous diagnoses.
  - Analysis of historical medical trends.
  - Longitudinal health tracking.

---

### 4. Agent Query Interface (ReAct Agent)

- A ReAct-style agent is implemented via `LangChain`'s `initialize_agent`.
- Supports intelligent **tool selection** and **contextual reasoning**.
- Users can ask:
  - “What diseases have I had in the past?”
  - “What were the diagnoses last year?”
- Agent intelligently decides:
  - Whether to search memory.
  - Whether to generate a fresh report.

---

### 5. Streamlit Interface

- Clean and interactive UI built using **Streamlit**.
- Main features:
  - Form-based patient metadata collection.
  - Image uploader with preview.
  - Real-time AI-generated report display.
  - View historical reports and memory logs.
  - Download report and memory data.
  - Agent chat window for querying medical history.

---


