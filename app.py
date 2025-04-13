import streamlit as st

# Configure Streamlit page - MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Notebook Analyzer",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with dark theme
st.markdown("""
    <style>
    .main {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    .stApp {
        background-color: #1E1E1E;
    }
    .explanation-block {
        background-color: #2D2D2D;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        color: #FFFFFF;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.5rem 1rem;
        border-radius: 5px;
        background-color: #2D2D2D;
        color: #FFFFFF;
    }
    .stTabs [aria-selected="true"] {
        background-color: #404040;
        color: white;
        border-left: 3px solid #808080;
    }
    .metric-card {
        background-color: #2D2D2D;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        color: #FFFFFF;
    }
    .code-block {
        background-color: #1E1E1E;
        color: #FFFFFF;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border: 1px solid #404040;
        white-space: pre-wrap;
        font-family: 'Courier New', Courier, monospace;
    }
    .stMarkdown {
        color: #FFFFFF;
    }
    .stTextInput>div>div>input {
        color: #FFFFFF;
        background-color: #2D2D2D;
    }
    .stButton>button {
        background-color: #404040;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #505050;
        transform: translateY(-2px);
    }
    .section-header {
        border-left: 3px solid #808080;
        padding-left: 10px;
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)

import nbformat
from nbformat import read
import json
import os
from typing import List, Dict, Any
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_extras.metric_cards import style_metric_cards
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from collections import defaultdict
import re

# Initialize NLTK - Download required data at startup
@st.cache_resource
def initialize_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        with st.spinner('Downloading required NLTK data...'):
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
    return True

# Initialize NLTK
initialize_nltk()

def extract_key_sentences(text: str, num_sentences: int = 3) -> str:
    """Extract key sentences from text using basic frequency analysis."""
    try:
        if not text or not isinstance(text, str):
            return "No content to analyze."

        # Clean and tokenize the text
        text = re.sub(r'#.*?\n', '', text)  # Remove Python comments
        text = re.sub(r'""".*?"""', '', text, flags=re.DOTALL)  # Remove docstrings
        text = text.strip()
        
        if not text:
            return "No content to analyze after cleaning."
        
        sentences = sent_tokenize(text)
        if not sentences:
            return text if text else "No content to analyze."
            
        # If we have fewer sentences than requested, return all of them
        if len(sentences) <= num_sentences:
            return ' '.join(sentences)
            
        try:
            # Calculate word frequencies
            stop_words = set(stopwords.words('english'))
            word_freq = defaultdict(int)
            
            for sentence in sentences:
                words = word_tokenize(sentence.lower())
                words = [word for word in words if word not in stop_words and word not in punctuation]
                for word in words:
                    word_freq[word] += 1
                    
            # Score sentences based on word frequencies
            sentence_scores = defaultdict(int)
            for i, sentence in enumerate(sentences):
                words = word_tokenize(sentence.lower())
                for word in words:
                    if word in word_freq:
                        sentence_scores[i] += word_freq[word]
                        
            # Get top sentences
            top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:num_sentences]
            top_sentences = sorted(top_sentences, key=lambda x: x[0])  # Sort by original order
            
            result = ' '.join(sentences[i] for i, _ in top_sentences)
            return result if result else "Could not extract meaningful sentences."
            
        except Exception as inner_e:
            st.error(f"Error in sentence analysis: {str(inner_e)}")
            return text[:500] + "..." if len(text) > 500 else text  # Fallback to original text
        
    except Exception as e:
        st.error(f"Error in text analysis: {str(e)}")
        return "Error analyzing text. Please try again."

def analyze_code(code: str) -> str:
    """Analyze Python code and explain it in simple, everyday language."""
    try:
        if not code or not isinstance(code, str):
            return "This cell is empty."

        # Clean the code
        code = re.sub(r'#.*?\n', '\n', code)
        code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
        code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)
        code = code.strip()
        
        if not code:
            return "This cell just has comments in it."
        
        explanation = []
        
        try:
            # Look for what the code is doing at a high level
            if 'print' in code:
                explanation.append("This code displays some information.")
            if 'return' in code:
                explanation.append("It gives back some results.")
            if '=' in code and not code.strip().startswith('def') and not code.strip().startswith('class'):
                explanation.append("It stores some data in variables.")
                
            # Check for data operations
            if any(x in code for x in ['.read', '.load', 'open(']):
                explanation.append("It reads data from files.")
            if any(x in code for x in ['.write', '.save', '.to_']):
                explanation.append("It saves data to files.")
            if any(x in code for x in ['pandas', 'df.', '.DataFrame']):
                explanation.append("It works with data tables.")
            if 'plot' in code or 'fig' in code:
                explanation.append("It creates some visualizations.")
                
            # Look for imports
            imports = re.findall(r'^(?:from|import)\s+([\w.]+)(?:\s+import\s+[\w.]+)?', code, re.MULTILINE)
            if imports:
                libraries = sorted(set(imports))
                if len(libraries) == 1:
                    explanation.append(f"It uses the {libraries[0]} package to help with its tasks.")
                elif len(libraries) <= 3:
                    explanation.append(f"It uses these packages to help: {', '.join(libraries)}.")
                else:
                    explanation.append(f"It uses several different packages including {', '.join(libraries[:3])} and others.")
            
            # Look for functions
            functions = re.findall(r'def\s+(\w+)\s*\([^)]*\)', code)
            if functions:
                if len(functions) == 1:
                    explanation.append(f"It creates a function named '{functions[0]}' that can be used later.")
                else:
                    explanation.append(f"It creates {len(functions)} functions that can be used later.")
            
            # Look for classes
            classes = re.findall(r'class\s+(\w+)[:\s]', code)
            if classes:
                if len(classes) == 1:
                    explanation.append(f"It creates a template (class) called '{classes[0]}' for organizing related code.")
                else:
                    explanation.append(f"It creates {len(classes)} templates for organizing code.")
            
            # Look for common patterns in simpler terms
            if '[' in code and 'for' in code and ']' in code:
                explanation.append("It processes a list of items all at once.")
            
            if 'try' in code:
                explanation.append("It has error handling to deal with potential problems.")
            
            if 'with' in code:
                explanation.append("It safely handles files or resources.")
            
            if 'if' in code:
                explanation.append("It makes decisions based on certain conditions.")
            
            if 'for' in code or 'while' in code:
                explanation.append("It repeats some actions multiple times.")
            
            # Add complexity description in simple terms
            lines = [line.strip() for line in code.split('\n') if line.strip()]
            if lines:
                if len(lines) < 5:
                    explanation.append("This is a very short and straightforward piece of code.")
                elif len(lines) < 10:
                    explanation.append("This is a simple piece of code that does a few things.")
                elif len(lines) < 20:
                    explanation.append("This is a medium-sized piece of code with several steps.")
                else:
                    explanation.append("This is a longer piece of code that does quite a few things.")
            
            if not explanation:
                return "This is a simple piece of code that does some basic operations."
            
            # Join everything in a natural way
            final_explanation = ' '.join(explanation)
            
            # Add a period if it doesn't end with one
            if not final_explanation.endswith('.'):
                final_explanation += '.'
                
            return final_explanation
            
        except Exception as inner_e:
            st.error(f"Error analyzing code patterns: {str(inner_e)}")
            return f"This is a piece of code with {len(code.split('\\n'))} lines."
        
    except Exception as e:
        st.error(f"Error in code analysis: {str(e)}")
        return "Sorry, I couldn't understand what this code does."

def get_cell_explanation(cell_content: str, cell_type: str) -> str:
    """Explain a notebook cell in simple terms."""
    try:
        if cell_type == 'code':
            return analyze_code(cell_content)
        else:
            # For text/markdown cells
            text = cell_content.strip()
            if not text:
                return "This cell is empty."
                
            # Look for headers/titles
            headers = re.findall(r'^#+\s+(.+)$', text, re.MULTILINE)
            if headers:
                if len(headers) == 1:
                    return f"This is a section about '{headers[0]}' with some additional explanation."
                else:
                    return f"This section has {len(headers)} main points: {', '.join(headers)}."
            
            # For regular text, give a simple summary
            summary = extract_key_sentences(text, num_sentences=2)
            return f"This text talks about: {summary}"
            
    except Exception as e:
        st.error(f"Error explaining cell: {str(e)}")
        return "Sorry, I couldn't understand what this cell does."

def get_notebook_summary(notebook) -> str:
    """Generate a clear, comprehensive summary of the notebook."""
    try:
        # Extract cells
        code_cells = [cell.source for cell in notebook.cells if cell.cell_type == 'code']
        markdown_cells = [cell.source for cell in notebook.cells if cell.cell_type == 'markdown']
        
        summary_parts = []
        
        # Analyze code content
        if code_cells:
            # Analyze imports
            all_imports = set()
            import_pattern = r'^(?:from|import)\s+([\w.]+)(?:\s+import\s+[\w.]+)?'
            for cell in code_cells:
                imports = re.findall(import_pattern, cell, re.MULTILINE)
                all_imports.update(imports)
            
            if all_imports:
                main_libraries = sorted(all_imports)
                summary_parts.append(f"📚 Main Libraries Used:\n" + "\n".join([f"• {lib}" for lib in main_libraries]))
            
            # Analyze functions and classes
            functions = []
            classes = []
            for cell in code_cells:
                functions.extend(re.findall(r'def\s+(\w+)\s*\([^)]*\)', cell))
                classes.extend(re.findall(r'class\s+(\w+)[:\s]', cell))
            
            if functions:
                summary_parts.append(f"⚙️ Functions Defined ({len(functions)}):\n" + 
                                  "\n".join([f"• {func}" for func in functions]))
            if classes:
                summary_parts.append(f"🏗️ Classes Defined ({len(classes)}):\n" + 
                                  "\n".join([f"• {cls}" for cls in classes]))
            
            # Analyze code complexity
            total_lines = sum(len(cell.split('\n')) for cell in code_cells)
            avg_cell_length = total_lines / len(code_cells)
            
            if avg_cell_length < 5:
                complexity = "Very simple"
            elif avg_cell_length < 10:
                complexity = "Simple"
            elif avg_cell_length < 20:
                complexity = "Moderate"
            else:
                complexity = "Complex"
            
            summary_parts.append(f"📊 Code Complexity Analysis:\n" +
                               f"• Average cell length: {avg_cell_length:.1f} lines\n" +
                               f"• Overall complexity: {complexity}\n" +
                               f"• Total lines of code: {total_lines}")
        
        # Analyze markdown content
        if markdown_cells:
            headers = []
            for cell in markdown_cells:
                headers.extend(re.findall(r'^#+\s+(.+)$', cell, re.MULTILINE))
            
            if headers:
                summary_parts.append(f"📝 Document Structure:\n" + "\n".join([f"• {header}" for header in headers[:5]]))
                if len(headers) > 5:
                    summary_parts.append(f"  ... and {len(headers) - 5} more sections")
        
        # Add notebook statistics
        total_cells = len(notebook.cells)
        code_cell_count = len(code_cells)
        markdown_cell_count = len(markdown_cells)
        
        stats = f"""📈 Notebook Overview:
• Total Cells: {total_cells}
• Code Cells: {code_cell_count} ({(code_cell_count/total_cells*100):.1f}%)
• Markdown Cells: {markdown_cell_count} ({(markdown_cell_count/total_cells*100):.1f}%)
• Code-to-Documentation Ratio: {code_cell_count/max(markdown_cell_count, 1):.1f}:1"""
        
        # Combine all parts with clear separation
        summary = "\n\n".join(summary_parts)
        
        return f"""🔍 Notebook Analysis Summary:

{summary}

{stats}"""
    except Exception as e:
        st.error(f"Error in get_notebook_summary: {str(e)}")
        return "Error generating summary. Please try again."

# Initialize session state
if 'notebook_content' not in st.session_state:
    st.session_state.notebook_content = None
if 'current_cell' not in st.session_state:
    st.session_state.current_cell = 0
if 'explanations' not in st.session_state:
    st.session_state.explanations = {}
if 'summary' not in st.session_state:
    st.session_state.summary = None
if 'analysis_metrics' not in st.session_state:
    st.session_state.analysis_metrics = None

# Sidebar configuration
with st.sidebar:
    st.title("📚 Notebook Analyzer")
    st.markdown("---")
    st.write("Upload and analyze your Jupyter notebooks")
    uploaded_file = st.file_uploader("", type=['ipynb'])
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This tool helps you analyze and understand your Jupyter notebooks by providing:
    - 📊 Detailed metrics and statistics
    - 📝 Cell-by-cell explanations
    - 📈 Visual analysis of notebook structure
    - 🔍 Code insights and suggestions
    """)

def process_notebook(file):
    try:
        notebook = read(file, as_version=4)
        return notebook
    except Exception as e:
        st.error(f"Error reading notebook: {str(e)}")
        return None

def analyze_notebook_metrics(notebook) -> Dict[str, Any]:
    """Analyze notebook metrics and statistics."""
    metrics = {
        "total_cells": len(notebook.cells),
        "code_cells": len([cell for cell in notebook.cells if cell.cell_type == 'code']),
        "markdown_cells": len([cell for cell in notebook.cells if cell.cell_type == 'markdown']),
        "avg_code_length": 0,
        "avg_markdown_length": 0,
        "imports": set(),
        "functions": [],
        "classes": []
    }
    
    total_code_length = 0
    total_markdown_length = 0
    
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            total_code_length += len(cell.source)
            # Extract imports
            for line in cell.source.split('\n'):
                if line.strip().startswith('import ') or line.strip().startswith('from '):
                    metrics["imports"].add(line.strip())
                elif line.strip().startswith('def '):
                    metrics["functions"].append(line.strip().split('def ')[1].split('(')[0])
                elif line.strip().startswith('class '):
                    metrics["classes"].append(line.strip().split('class ')[1].split(':')[0])
        elif cell.cell_type == 'markdown':
            total_markdown_length += len(cell.source)
    
    metrics["avg_code_length"] = total_code_length / metrics["code_cells"] if metrics["code_cells"] > 0 else 0
    metrics["avg_markdown_length"] = total_markdown_length / metrics["markdown_cells"] if metrics["markdown_cells"] > 0 else 0
    
    return metrics

# Main content area
st.title("📚 Notebook Analyzer")

if uploaded_file is not None:
    if st.session_state.notebook_content is None:
        st.session_state.notebook_content = process_notebook(uploaded_file)
        st.session_state.explanations = {}
        st.session_state.summary = None
        st.session_state.analysis_metrics = None
    
    if st.session_state.notebook_content:
        # Display notebook metrics
        if st.session_state.analysis_metrics is None:
            st.session_state.analysis_metrics = analyze_notebook_metrics(st.session_state.notebook_content)
        
        # Create tabs for different analysis views
        tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔍 Cell Analysis", "📈 Metrics"])
        
        with tab1:
            # Display notebook summary
            st.header("📝 Notebook Summary")
            if st.session_state.summary is None:
                with st.spinner("Generating notebook summary..."):
                    st.session_state.summary = get_notebook_summary(st.session_state.notebook_content)
            
            st.markdown(f"""
                <div class='explanation-block'>
                    {st.session_state.summary}
                </div>
            """, unsafe_allow_html=True)
            
            # Display key metrics
            st.header("📊 Key Metrics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                    <div class='metric-card'>
                        <h3>Total Cells</h3>
                        <h2>{st.session_state.analysis_metrics["total_cells"]}</h2>
                    </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                    <div class='metric-card'>
                        <h3>Code Cells</h3>
                        <h2>{st.session_state.analysis_metrics["code_cells"]}</h2>
                    </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                    <div class='metric-card'>
                        <h3>Markdown Cells</h3>
                        <h2>{st.session_state.analysis_metrics["markdown_cells"]}</h2>
                    </div>
                """, unsafe_allow_html=True)
        
        with tab2:
            # Cell-by-cell analysis
            st.markdown("<h2 class='section-header'>🔍 Cell-by-Cell Analysis</h2>", unsafe_allow_html=True)
            
            # Add a button to analyze the entire notebook
            if st.button("🔍 Analyze Entire Notebook", use_container_width=True):
                with st.spinner("Analyzing notebook..."):
                    # Generate full notebook analysis
                    full_analysis = get_notebook_summary(st.session_state.notebook_content)
                    st.markdown(f"<div class='explanation-block'>{full_analysis}</div>", unsafe_allow_html=True)
                    
                    st.markdown("<h3 class='section-header'>📝 Detailed Cell Analysis</h3>", unsafe_allow_html=True)
                    for idx, cell in enumerate(st.session_state.notebook_content.cells):
                        with st.expander(f"Cell {idx + 1} ({cell.cell_type})"):
                            # Show cell content
                            if cell.cell_type == 'code':
                                st.code(cell.source, language='python')
                            else:
                                st.markdown(f"<div class='code-block'>{cell.source}</div>", unsafe_allow_html=True)
                            
                            # Generate or get explanation
                            if idx not in st.session_state.explanations:
                                st.session_state.explanations[idx] = get_cell_explanation(
                                    cell.source,
                                    cell.cell_type
                                )
                            st.markdown(f"<div class='explanation-block'>{st.session_state.explanations[idx]}</div>", 
                                      unsafe_allow_html=True)
                st.success("Analysis complete! 🎉")
            
            st.markdown("<h3 class='section-header'>📝 Individual Cell Analysis</h3>", unsafe_allow_html=True)
            
            # Create two columns for cell content and explanation
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Cell Content")
                cells = st.session_state.notebook_content.cells
                if cells:
                    cell_index = st.slider("Select Cell", 0, len(cells)-1, st.session_state.current_cell)
                    st.session_state.current_cell = cell_index
                    
                    cell = cells[cell_index]
                    # Use st.code instead of markdown for code display
                    if cell.cell_type == 'code':
                        st.code(cell.source, language='python')
                    else:
                        st.markdown(f"""
                            <div class='code-block'>
                                {cell.source}
                            </div>
                        """, unsafe_allow_html=True)
            
            with col2:
                st.subheader("Analysis")
                if cells:
                    cell = cells[st.session_state.current_cell]
                    if cell_index not in st.session_state.explanations:
                        with st.spinner("Generating analysis..."):
                            st.session_state.explanations[cell_index] = get_cell_explanation(
                                cell.source,
                                cell.cell_type
                            )
                    st.markdown(f"""
                        <div class='explanation-block'>
                            {st.session_state.explanations[cell_index]}
                        </div>
                    """, unsafe_allow_html=True)
        
        with tab3:
            st.header("📈 Detailed Metrics")
            
            # Create two columns for metrics
            col1, col2 = st.columns(2)
            
            with col1:
                # Display imports
                st.subheader("📦 Imports")
                imports_text = '\n'.join(sorted(st.session_state.analysis_metrics["imports"]))
                st.code(imports_text, language="python")
                
                # Display functions
                st.subheader("⚙️ Functions")
                st.markdown(f"""
                    <div class='explanation-block'>
                        {', '.join(st.session_state.analysis_metrics["functions"])}
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Display classes
                st.subheader("🏗️ Classes")
                st.markdown(f"""
                    <div class='explanation-block'>
                        {', '.join(st.session_state.analysis_metrics["classes"])}
                    </div>
                """, unsafe_allow_html=True)
                
                # Create visualizations
                st.subheader("📊 Cell Distribution")
                cell_types = ['Code', 'Markdown']
                cell_counts = [st.session_state.analysis_metrics["code_cells"], 
                              st.session_state.analysis_metrics["markdown_cells"]]
                
                fig = go.Figure(data=[go.Pie(
                    labels=cell_types,
                    values=cell_counts,
                    hole=.3,
                    marker_colors=['#4b8bb5', '#f0f2f6']
                )])
                fig.update_layout(
                    title='Cell Type Distribution',
                    showlegend=True,
                    height=400,
                    paper_bgcolor='#1E1E1E',
                    plot_bgcolor='#1E1E1E',
                    font=dict(color='#FFFFFF')
                )
                st.plotly_chart(fig, use_container_width=True)
else:
    st.info("👆 Please upload a Jupyter notebook to begin analysis.") 