"""
Agentic Tabular Data Analytics Platform
Main Streamlit Application
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from agents_module import PlannerAgent, CodeGeneratorAgent, VerifierAgent, ExecutionAgent, ExplainerAgent
from utils_module import load_data, get_data_info

# Page configuration
st.set_page_config(
    page_title="Agentic Analytics Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .agent-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .agent-planner { background-color: #e3f2fd; border-left: 4px solid #2196f3; }
    .agent-codegen { background-color: #f3e5f5; border-left: 4px solid #9c27b0; }
    .agent-verifier { background-color: #e8f5e9; border-left: 4px solid #4caf50; }
    .agent-executor { background-color: #fff3e0; border-left: 4px solid #ff9800; }
    .agent-explainer { background-color: #fce4ec; border-left: 4px solid #e91e63; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'data' not in st.session_state:
    st.session_state.data = None
if 'data_info' not in st.session_state:
    st.session_state.data_info = None

# Initialize agents
@st.cache_resource
def initialize_agents():
    return {
        'planner': PlannerAgent(),
        'code_generator': CodeGeneratorAgent(),
        'verifier': VerifierAgent(),
        'executor': ExecutionAgent(),
        'explainer': ExplainerAgent()
    }

agents = initialize_agents()

# Header
st.markdown('<p class="main-header">📊 Agentic Analytics Platform</p>', unsafe_allow_html=True)
st.markdown("Upload your dataset and ask questions in natural language")
st.divider()

# Sidebar - File Upload
with st.sidebar:
    st.header("📁 Data Upload")
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload your tabular dataset"
    )
    
    if uploaded_file is not None:
        with st.spinner("Loading data..."):
            df = load_data(uploaded_file)
            if df is not None:
                st.session_state.data = df
                st.session_state.data_info = get_data_info(df)
                st.success("✅ Data loaded successfully!")
    
    # Data Preview
    if st.session_state.data is not None:
        st.divider()
        st.header("📊 Data Overview")
        
        info = st.session_state.data_info
        st.metric("Rows", f"{info['rows']:,}")
        st.metric("Columns", info['columns'])
        
        with st.expander("📋 Column Types"):
            for col, dtype in info['column_types'].items():
                st.text(f"{col}: {dtype}")
        
        with st.expander("👁️ Preview (First 5 rows)"):
            st.dataframe(st.session_state.data.head(), use_container_width=True)
        
        with st.expander("📈 Quick Stats"):
            st.dataframe(st.session_state.data.describe(), use_container_width=True)
    
    st.divider()
    st.header("ℹ️ About")
    st.info("""
    **Agentic AI System**
    
    This platform uses 5 specialized AI agents:
    - 🧠 Planner: Interprets queries
    - 💻 Code Generator: Creates analysis code
    - 🛡️ Verifier: Ensures code safety
    - ⚡ Executor: Runs analysis
    - 📝 Explainer: Provides insights
    """)

# Main Chat Interface
col1, col2 = st.columns([2, 1])

with col1:
    st.header("💬 Analysis Chat")
    
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            if msg['role'] == 'user':
                with st.chat_message("user"):
                    st.write(msg['content'])
            elif msg['role'] == 'system':
                with st.chat_message("assistant"):
                    st.success(msg['content'])
            elif msg['role'] == 'agent':
                agent_type = msg.get('agent', 'assistant')
                with st.chat_message("assistant"):
                    st.markdown(f"<div class='agent-box agent-{agent_type}'>{msg['content']}</div>", 
                              unsafe_allow_html=True)
            elif msg['role'] == 'result':
                with st.chat_message("assistant"):
                    st.markdown(msg['content'])
                    if 'figure' in msg and msg['figure'] is not None:
                        st.pyplot(msg['figure'])
                    if 'output' in msg and msg['output']:
                        with st.expander("📋 Detailed Output"):
                            st.code(msg['output'])
    
    # Example queries
    if st.session_state.data is not None and len(st.session_state.messages) == 0:
        st.info("**Try these example queries:**")
        example_cols = st.columns(3)
        with example_cols[0]:
            if st.button("📊 Describe dataset"):
                query = "Describe my dataset"
            else:
                query = None
        with example_cols[1]:
            if st.button("🔗 Show correlations"):
                query = "Show me correlations between variables"
            else:
                query = query if query else None
        with example_cols[2]:
            if st.button("🎯 Find outliers"):
                query = "Find outliers in my data"
            else:
                query = query if query else None
    else:
        query = None
    
    # Chat input
    user_query = st.chat_input("Ask a question about your data...", disabled=st.session_state.data is None)
    
    if query:
        user_query = query
    
    # Process query
    if user_query:
        # Add user message
        st.session_state.messages.append({
            'role': 'user',
            'content': user_query
        })
        
        # Process through agent pipeline
        with st.spinner("🤖 AI Agents are working..."):
            try:
                # Agent 1: Planner
                plan = agents['planner'].plan(user_query, st.session_state.data_info)
                st.session_state.messages.append({
                    'role': 'agent',
                    'agent': 'planner',
                    'content': f"**🧠 Planner Agent**\n\n" + 
                              f"**Analysis Type:** {plan['plan_type']}\n\n" +
                              f"**Steps:**\n" + "\n".join([f"{i+1}. {step}" for i, step in enumerate(plan['steps'])])
                })
                
                # Agent 2: Code Generator
                code_gen = agents['code_generator'].generate(plan, st.session_state.data_info)
                st.session_state.messages.append({
                    'role': 'agent',
                    'agent': 'codegen',
                    'content': f"**💻 Code Generator Agent**\n\n" +
                              f"Generated {len(code_gen['code'].splitlines())} lines of Python code\n\n" +
                              f"**Libraries:** {', '.join(code_gen['libraries'])}"
                })
                
                # Agent 3: Verifier
                verification = agents['verifier'].verify(code_gen['code'])
                if verification['safe']:
                    st.session_state.messages.append({
                        'role': 'agent',
                        'agent': 'verifier',
                        'content': f"**🛡️ Verifier Agent**\n\n✅ Code passed all security checks\n✅ No dangerous operations detected"
                    })
                    
                    # Agent 4: Executor
                    execution = agents['executor'].execute(code_gen['code'], st.session_state.data)
                    
                    if execution['success']:
                        st.session_state.messages.append({
                            'role': 'agent',
                            'agent': 'executor',
                            'content': f"**⚡ Executor Agent**\n\n✅ Code executed successfully"
                        })
                        
                        # Agent 5: Explainer
                        explanation = agents['explainer'].explain(execution, plan)
                        st.session_state.messages.append({
                            'role': 'result',
                            'content': explanation['explanation'],
                            'output': execution.get('output', ''),
                            'figure': execution.get('figure')
                        })
                    else:
                        st.session_state.messages.append({
                            'role': 'agent',
                            'agent': 'executor',
                            'content': f"**❌ Execution Error**\n\n{execution['error']}"
                        })
                else:
                    st.session_state.messages.append({
                        'role': 'agent',
                        'agent': 'verifier',
                        'content': f"**⚠️ Security Issues Detected**\n\n" + "\n".join(verification['issues'])
                    })
                    
            except Exception as e:
                st.session_state.messages.append({
                    'role': 'system',
                    'content': f"❌ Error: {str(e)}"
                })
        
        st.rerun()

with col2:
    st.header("🎯 Agent Status")
    
    if st.session_state.data is not None:
        st.success("✅ Data Loaded")
    else:
        st.warning("⏳ Waiting for data")
    
    st.divider()
    
    # Agent cards
    agents_info = [
        ("🧠", "Planner", "Interprets queries", "planner"),
        ("💻", "Code Generator", "Creates analysis code", "codegen"),
        ("🛡️", "Verifier", "Ensures safety", "verifier"),
        ("⚡", "Executor", "Runs analysis", "executor"),
        ("📝", "Explainer", "Provides insights", "explainer")
    ]
    
    for icon, name, desc, agent_type in agents_info:
        with st.container():
            st.markdown(f"""
            <div class='agent-box agent-{agent_type}'>
                <h4>{icon} {name}</h4>
                <p style='margin:0; color: #666;'>{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.divider()
    
    if st.button("🗑️ Clear Chat History", type="secondary", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p><strong>Agentic Analytics Platform</strong> | Powered by Multi-Agent AI System</p>
    <p style='font-size: 0.9rem;'>Safe • Transparent • Intelligent</p>
</div>
""", unsafe_allow_html=True)