# 🤖 Agentic Tabular Data Analytics Platform

A complete multi-agent AI system for analyzing tabular data using natural language queries. Built with Python, Streamlit, and a modular agent architecture.

## 🎯 Features

### **5 Specialized AI Agents**
1. **🧠 Planner Agent** - Interprets natural language queries and creates analytical plans
2. **💻 Code Generator Agent** - Generates safe, executable Python code for analysis
3. **🛡️ Verifier Agent** - Ensures code safety with pattern matching and AST parsing
4. **⚡ Executor Agent** - Runs code in a sandboxed environment with restricted globals
5. **📝 Explainer Agent** - Converts technical results into clear, actionable insights

### **Analytics Capabilities**
- **Descriptive Statistics**: Mean, median, std, missing values, distributions
- **Correlation Analysis**: Relationship detection with visual heatmaps
- **Data Visualization**: Multi-panel dashboards with various chart types
- **Outlier Detection**: IQR-based anomaly identification
- **Predictive Modeling**: Linear regression with performance metrics
- **Group-By Analysis**: Categorical aggregations with visualizations

### **Security Features**
- Pattern-based dangerous code detection
- Library whitelisting (pandas, numpy, matplotlib, sklearn, seaborn)
- AST-based import verification
- Sandboxed execution environment
- No file system or network access

## 📁 Project Structure

```
agentic-analytics/
│
├── app.py                 # Main Streamlit application
├── agents.py              # All 5 AI agents implementation
├── utils.py               # Data loading and processing utilities
├── requirements.txt       # Python dependencies
├── README.md             # This file
│
└── data/                 # (Optional) Sample datasets
    └── sample.csv
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or create project directory
mkdir agentic-analytics
cd agentic-analytics

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### 3. Usage

1. **Upload Data**: Click "Choose a CSV or Excel file" in the sidebar
2. **View Overview**: Check the data preview, column types, and statistics
3. **Ask Questions**: Type natural language queries like:
   - "Describe my dataset"
   - "Show me correlations between variables"
   - "Find outliers in my data"
   - "Create visualizations"
   - "Build a predictive model"
4. **Review Results**: See agent workflow, visualizations, and insights

## 📊 Example Queries

### Descriptive Analysis
```
"Give me a summary of my dataset"
"Show basic statistics"
"What's the overview of this data?"
```

### Correlation Analysis
```
"Show me correlations"
"Which variables are related?"
"Find relationships in the data"
```

### Visualization
```
"Visualize my data"
"Create charts for this dataset"
"Show me plots"
```

### Outlier Detection
```
"Find outliers"
"Detect anomalies"
"Show unusual values"
```

### Predictive Modeling
```
"Build a predictive model"
"Create a regression model"
"Predict values"
```

### Group-By Analysis
```
"Group by category"
"Show aggregated results"
"Break down by segments"
```

## 🔧 Configuration

### Customizing Agents

Edit `agents.py` to modify agent behavior:

```python
# Add new analysis types to PlannerAgent
self.plan_templates['new_analysis'] = {
    'keywords': ['custom', 'special'],
    'steps': ['Step 1', 'Step 2'],
    'tools': ['pandas.custom()'],
    'code_type': 'new_analysis'
}

# Add code generation template to CodeGeneratorAgent
def _generate_custom_code(self):
    return '''
    # Your custom analysis code
    print("Custom analysis")
    result = {'custom': 'data'}
    '''
```

### Adjusting Security Settings

Modify `VerifierAgent` in `agents.py`:

```python
# Add more allowed libraries
self.allowed_imports = [
    'pandas', 'numpy', 'matplotlib', 'seaborn',
    'sklearn', 'scipy', 'statsmodels',
    'your_custom_library'  # Add here
]

# Add patterns to block
self.dangerous_patterns = [
    'dangerous_function',
    # Add more patterns
]
```

### Data Size Limits

Adjust limits in `utils.py`:

```python
def validate_dataframe(df: pd.DataFrame, max_rows: int = 1000000):
    # Change max_rows for larger datasets
    pass

def sample_dataframe(df: pd.DataFrame, max_rows: int = 10000):
    # Adjust sampling threshold
    pass
```

## 🔌 LLM Integration (Optional)

To use real LLM for more sophisticated planning and explanation:

### Option 1: Ollama (Local)

```bash
# Install Ollama
# Visit https://ollama.ai

# Pull a model
ollama pull llama3.2

# Modify agents.py to use Ollama API
import requests

def llm_call(prompt):
    response = requests.post('http://localhost:11434/api/generate',
        json={'model': 'llama3.2', 'prompt': prompt})
    return response.json()
```

### Option 2: HuggingFace Transformers

```bash
pip install transformers torch

# In agents.py
from transformers import pipeline

generator = pipeline('text-generation', model='gpt2')
response = generator(prompt, max_length=200)
```

## 🛡️ Security Considerations

### Current Protections
- ✅ No file system access (no `open()`, `file()`)
- ✅ No OS commands (no `os.system()`, `subprocess`)
- ✅ No network access (no `requests`, `urllib`)
- ✅ No code injection (no `eval()`, `exec()` from user input)
- ✅ Library whitelisting
- ✅ AST-based import verification

### Additional Recommendations for Production
1. Run in Docker container with resource limits
2. Implement user authentication and rate limiting
3. Use separate process/container for code execution
4. Add execution timeouts (currently basic)
5. Log all code execution for audit
6. Implement more sophisticated AST analysis

## 📦 Dependencies

### Core Libraries
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **matplotlib**: Plotting and visualization
- **seaborn**: Statistical data visualization
- **scikit-learn**: Machine learning models
- **streamlit**: Web interface

### Optional Libraries
- **duckdb**: For large dataset queries
- **polars**: High-performance DataFrames
- **ollama**: Local LLM integration
- **transformers**: HuggingFace models

## 🎨 Customization Examples

### Adding a New Analysis Type

1. **Update PlannerAgent** (`agents.py`):
```python
'time_series': {
    'keywords': ['time', 'trend', 'seasonal'],
    'steps': ['Detect time column', 'Plot trend', 'Analyze seasonality'],
    'tools': ['df.plot()', 'seasonal_decompose()'],
    'code_type': 'time_series'
}
```

2. **Add Code Generation** (`agents.py`):
```python
def _generate_time_series_code(self):
    return '''
    # Time series analysis
    import matplotlib.pyplot as plt
    
    # Your code here
    result = {'trend': 'upward'}
    '''
```

3. **Add Explanation** (`agents.py`):
```python
def _explain_time_series(self, result):
    return "📈 Time series analysis shows..."
```

### Custom Styling

Modify CSS in `app.py`:
```python
st.markdown("""
<style>
    .custom-class {
        background-color: #your-color;
        /* Your styles */
    }
</style>
""", unsafe_allow_html=True)
```

## 🐛 Troubleshooting

### Common Issues

**1. Module Import Errors**
```bash
# Ensure virtual environment is activated
pip install -r requirements.txt --upgrade
```

**2. Data Upload Fails**
```python
# Check file encoding in utils.py
df = pd.read_csv(uploaded_file, encoding='utf-8')  # Try different encodings
```

**3. Code Execution Errors**
```python
# Check ExecutionAgent timeout and memory limits
# Increase timeout in agents.py if needed
```

**4. Visualization Not Showing**
```python
# Ensure matplotlib backend is set correctly
plt.switch_backend('Agg')
```

## 📈 Performance Tips

1. **Large Datasets**: Use sampling in `utils.py`
2. **Memory Issues**: Consider DuckDB for out-of-core processing
3. **Slow Execution**: Cache agent initialization with `@st.cache_resource`
4. **Network Lag**: Run locally instead of cloud deployment

## 🤝 Contributing

To extend this platform:

1. Add new agent types in `agents.py`
2. Extend code generation templates
3. Add more security patterns
4. Improve LLM integration
5. Add more analytics capabilities

## 📄 License

This project is provided as-is for educational purposes. Modify and use as needed.

## 🔮 Future Enhancements

- [ ] Multi-file dataset support
- [ ] SQL query generation
- [ ] Advanced ML models (Random Forest, XGBoost)
- [ ] Interactive chart editing
- [ ] Export reports to PDF
- [ ] Collaborative features
- [ ] Real-time data streaming
- [ ] Natural language to SQL
- [ ] Advanced time series analysis
- [ ] A/B testing capabilities

## 💡 Tips

- Start with small datasets to test functionality
- Use example queries to understand capabilities
- Check agent workflow to see how analysis is performed
- Review generated code in detailed output
- Export visualizations by right-clicking charts

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review agent logs in detailed output
3. Verify data format and size
4. Test with sample datasets first

---

**Built with ❤️ using Python, Streamlit, and Multi-Agent AI Architecture**