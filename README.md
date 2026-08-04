# Talk2Tables
# 🤖 Agentic Analytics Platform

A **secure, multi-agent AI platform** for analyzing tabular data utilizing **natural language**. Designed with modular agents, strong security guarantees, and an interactive Streamlit interface.

> Upload your dataset, ask questions in plain English, and get **insights, code, and visualizations** — automatically.

---



![DataAI](https://www.datagaps.com/wp-content/uploads/Agentic-AI-for-Data-Analytics-Validations-1024x536.jpg)


## ✨ Key Features

### 🧠 Multi-Agent Architecture

The system is powered by **5 specialized AI agents**, each responsible for a distinct role:

1. 🧠 **Planner Agent** – Understands client intent and designs an analysis plan
2. 💻 **Code Generator Agent** – Produces safe, executable Python code
3. 🛡️ **Verifier Agent** – Validates code using pattern checks & AST analysis
4. ⚡ **Executor Agent** – Runs code in a sandboxed environment
5. 📝 **Explainer Agent** – Translates results into human-friendly insights

---

### 📊 Analytics Capabilities

* 📈 Descriptive Statistics (mean, median, std, missing values)
* 🔗 Correlation Analysis with heatmaps
* 📊 Interactive Visualizations (multiple chart types)
* 🚨 Outlier Detection (IQR-based)
* 🤖 Predictive Modeling (Linear Regression)
* 🧩 Group-By & Aggregation Analysis

---

### 🛡️ Security by Design

* ✅ Library whitelisting
* ✅ AST-based import verification
* ✅ Sandboxed execution environment

Built with **safety-first principles** to enable responsible AI-based analytics.

---

## 🖥️ Demo Workflow

1. 📤 Upload a CSV or Excel file
2. 🔍 Preview data & statistics
3. 🗣️ Ask questions in natural language
4. 🧠 Watch agents plan, generate & verify code
5. 📊 View charts, tables & explanations

---

## 🚀 Quick Start

### 1️⃣ Installation

```bash
# Clone repository
git clone https://github.com/MohammadMaftoun/Talk2Tables.git
cd agentic-analytics

# Create virtual environment
python -m venv venv

# Activate environment
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

### 2️⃣ Run the App

```bash
streamlit run app.py
```

The app will be available at:
👉 `http://localhost:8501`

---

## 🧪 Example Questions

### 📊 Descriptive Analysis

* "Give me a summary of the dataset"
* "Show basic statistics"

### 🔗 Correlation

* "Which variables are correlated?"
* "Show correlation heatmap"

### 📈 Visualization

* "Visualize my data"
* "Create charts for all numeric columns"

### 🚨 Outliers

* "Find outliers in the dataset"
* "Detect anomalies"

### 🤖 Predictive Modeling

* "Build a regression model"
* "Predict values based on features"

---

## 📁 Project Structure

```text
agentic-analytics/
│
├── app.py              # Streamlit application
├── agents.py           # All AI agents
├── utils.py            # Data utilities & validation
├── requirements.txt    # Dependencies
├── README.md           # Documentation
│
└── data/               # Optional sample datasets
```

---

## 🔧 Customization

### ➕ Add New Analysis Types

Extend the **Planner Agent**:

```python
self.plan_templates['time_series'] = {
    'keywords': ['time', 'trend'],
    'steps': ['Detect time column', 'Analyze trend'],
    'tools': ['pandas', 'matplotlib'],
    'code_type': 'time_series'
}
```

Add code generation logic in `CodeGeneratorAgent` and explanations in `ExplainerAgent`.

---

## 🔌 Optional LLM Integration

The platform can be connected to real LLMs:

* 🧠 **Ollama (Local Models)**
* 🤗 **HuggingFace Transformers**

This enables deeper reasoning, better explanations, and richer plans.

---

## ⚙️ Performance Tips

* 📉 Use sampling for large datasets
* 🐘 Use DuckDB for big data workloads
* 💾 Cache agent initialization
* 🧪 Start with small datasets

---

## 🧩 Use Cases

* 📊 Data Exploration & EDA
* 🧠 AI-assisted Analytics
* 🎓 Education & Teaching Data Science
* 🧪 Prototyping ML Pipelines
* 🏢 Internal BI Tools

---

## 🤝 Contributing

Contributions are welcome!

* Add new agents
* Improve security checks
* Extend analytics capabilities
* Enhance UI/UX

---

## 📄 License

Provided for educational and experimental use. Customize voluntarily.

---

## 🌟 Roadmap

* [ ] Advanced ML Models (RF, XGBoost)
* [ ] SQL & NL-to-SQL
* [ ] PDF Report Export
* [ ] Multi-file Datasets
* [ ] Real-time Data Streams

---

**Built with ❤️ using Python, Streamlit, and Multi-Agent AI** 🚀
