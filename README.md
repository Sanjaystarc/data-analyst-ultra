---
title: DataMind Agent
emoji: 🧠
colorFrom: purple
colorTo: blue
sdk: streamlit
sdk_version: "1.40.1"
python_version: "3.10"
app_file: app.py
pinned: false
---

<div align="center">

# 🧠 DataMind Agent

### Autonomous AI Data Analyst — LangChain + Google Gemini + Streamlit

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.3.7-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)](https://langchain.com)
[![Gemini](https://img.shields.io/badge/Google_Gemini-2.5_Flash-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://aistudio.google.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40.1-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![HuggingFace](https://img.shields.io/badge/Deployed-HuggingFace_Spaces-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**Upload any data file → Ask a question → Watch the agent think, plan, and deliver insights autonomously**

[🚀 Live Demo](#-live-demo) • [⚡ Quick Start](#-quick-start) • [🔧 How It Works](#-how-it-works) • [🛠️ Tech Stack](#️-tech-stack)

</div>

---

## 🎯 What is DataMind Agent?

DataMind Agent is a **production-grade Agentic AI system** that autonomously analyzes any dataset. Unlike traditional AI chatbots that respond in a single step, DataMind operates as a true autonomous agent:

```
You give a goal
      ↓
Agent THINKS & PLANS
      ↓
Selects the right tools → Executes them in sequence
      ↓
Reflects on results → Self-corrects if needed
      ↓
Synthesizes a clear, insightful answer
      ↓
Shows you its full reasoning chain 🔍
```

> **Regular AI** = You ask → It answers. One shot.
>
> **DataMind Agent** = You give a goal → It plans, acts, checks, retries — until the job is done.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🤖 **True Agentic AI** | Autonomous multi-step reasoning loop — not just a chatbot |
| 🔧 **7 Specialized Tools** | Agent picks the right tool automatically for each task |
| 🧠 **Conversation Memory** | Remembers context across the entire session |
| 🔍 **Transparent Reasoning** | See every tool the agent called and why |
| 📊 **Smart Visualizations** | Agent recommends and renders the best chart for your question |
| 📂 **Multi-Format Support** | CSV, Excel (.xlsx/.xls), JSON |
| 🔁 **Self-Correction** | Retries with a different approach if a tool fails |
| 🆓 **Free to Run** | Uses Google Gemini free tier — 1,500 requests/day |

---

## 🚀 Live Demo

> 🔗 **[Try it on Hugging Face Spaces](#)**
>
> Just bring your own free [Gemini API key](https://aistudio.google.com/app/apikey) — no setup needed!

---

## 🔧 How It Works

### The Agentic Loop

```
+-------------------------------------------------------+
|                    DATAMIND AGENT                     |
+-------------------------------------------------------+
|                                                       |
|   User Question                                       |
|        |                                              |
|        v                                              |
|   +---------+    +--------------------------------+   |
|   |  THINK  |───>|  Which tools do I need?        |   |
|   +---------+    |  What order should I use?      |   |
|        |         +--------------------------------+   |
|        v                                              |
|   +---------+    +--------------------------------+   |
|   |   ACT   |───>|  [1] profile_data              |   |
|   +---------+    |  [2] analyze_column            |   |
|        |         |  [3] find_correlations         |   |
|        v         |  [4] detect_anomalies          |   |
|   +---------+    |  [5] run_aggregation           |   |
|   | REFLECT |───>|  [6] generate_insight_report   |   |
|   +---------+    |  [7] recommend_chart           |   |
|        |         +--------------------------------+   |
|        v                                              |
|   +----------+                                        |
|   |SYNTHESIZE|───>  Final Answer + Chart              |
|   +----------+                                        |
|                                                       |
+-------------------------------------------------------+
```

### The 7 Agent Tools

| Tool | What It Does |
|---|---|
| `profile_data` | Full dataset overview — shape, types, nulls, stats |
| `analyze_column` | Deep statistical analysis of any single column |
| `find_correlations` | Pearson correlation matrix with insight highlights |
| `detect_anomalies` | IQR-based outlier detection across all numeric columns |
| `run_aggregation` | Autonomous group-by calculations (sum, mean, count...) |
| `generate_insight_report` | Full multi-section report with data quality score |
| `recommend_chart` | Picks best visualization type for any question |

---

## 🛠️ Tech Stack

| Layer | Technology | Version | Purpose |
|---|---|---|---|
| **LLM** | Google Gemini 2.5 Flash | Latest | Reasoning & language understanding |
| **Agent Framework** | LangChain | 0.3.7 | Tool calling, agent loop, memory |
| **Agent Type** | Tool Calling Agent | — | Autonomous multi-step execution |
| **Memory** | LangChain Conversation Memory | — | Context retention across questions |
| **UI** | Streamlit | 1.40.1 | Interactive web interface |
| **Data Engine** | Pandas | 2.x | File parsing, profiling, aggregations |
| **Visualization** | Plotly | 5.x | Interactive charts (9 types) |
| **File Support** | OpenPyXL + xlrd | — | Excel parsing |
| **Deployment** | Hugging Face Spaces | — | Free cloud hosting |
| **Language** | Python | 3.10 | Core language |

---

## ⚡ Quick Start

### Option A — Use the Live Demo
Visit the [Hugging Face Space](#), enter your free Gemini API key, upload a file, and start chatting!

### Option B — Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/datamind-agent
cd datamind-agent
```

**2. Create virtual environment**
```bash
python -m venv venv

# Mac/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add your Gemini API key**
```bash
echo "GOOGLE_API_KEY=AIzaSyYOUR_KEY_HERE" > .env
```
> Get your free key at [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

**5. Run the app**
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) 🎉

---

## 💬 Example Questions to Ask

```
# Single-tool questions
"Give me an overview of this dataset"
"Analyze the sales column in detail"
"Are there any outliers or anomalies?"
"What is the total revenue by category?"

# Multi-tool questions (agent chains tools automatically)
"Find outliers in profit, then show me which region performs best"
"Give me a full insight report with quality score and recommendations"
"Analyze all numeric columns and tell me what correlates with sales"

# Memory test (ask in sequence)
"What is the average sales?"
"Which region has the highest of that?"   ← agent remembers context
"How does that compare to profit?"         ← agent keeps the thread
```

---

## 📁 Project Structure

```
datamind-agent/
├── app.py              # Streamlit UI — 4 tabs, chat interface
├── core_agent.py       # LangChain agent, 7 tools, memory
├── requirements.txt    # Python dependencies
├── sample_data.csv     # Test dataset (30-row sales data)
├── .env                # API key (never commit this!)
└── README.md           # This file
```

---

## 🆚 vs Regular AI App

| | Regular AI App | DataMind Agent |
|---|---|---|
| Architecture | Single LLM call | Agent loop |
| Tools | ❌ None | ✅ 7 specialized tools |
| Memory | ❌ No | ✅ Full conversation memory |
| Multi-step reasoning | ❌ No | ✅ Yes |
| Self-correction | ❌ No | ✅ Yes |
| Transparency | ❌ Black box | ✅ Full reasoning visible |
| True Agentic AI | ❌ No | ✅ Yes |

---

## 🔐 Security

- API keys are never stored — entered per session only
- Use Hugging Face Secrets or `.env` file locally
- Never commit your `.env` file — it's in `.gitignore`

---

## 🗺️ Roadmap

- [ ] PDF file support
- [ ] SQL database connection
- [ ] Export full report to PDF
- [ ] Multi-file comparison mode
- [ ] Deploy to Streamlit Cloud

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

<div align="center">

**Built with ❤️ using LangChain + Google Gemini + Streamlit**

⭐ Star this repo if you found it useful!

</div>
