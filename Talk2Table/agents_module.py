"""
AI Agents for Tabular Data Analytics
Contains all 5 specialized agents
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import io
import sys
from datetime import datetime
from typing import Dict, List, Any

# Import sklearn components
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


class PlannerAgent:
    """Agent 1: Interprets user queries and creates analytical plans"""
    
    def __init__(self):
        self.plan_templates = {
            'descriptive': {
                'keywords': ['describe', 'summary', 'statistics', 'overview', 'info', 'basic'],
                'steps': [
                    'Load and inspect data structure',
                    'Calculate descriptive statistics',
                    'Identify missing values',
                    'Analyze data types'
                ],
                'tools': ['pandas.describe()', 'pandas.info()', 'isnull().sum()'],
                'code_type': 'descriptive'
            },
            'correlation': {
                'keywords': ['correlation', 'relationship', 'relate', 'corr', 'associated'],
                'steps': [
                    'Select numeric columns',
                    'Calculate correlation matrix',
                    'Identify strong correlations',
                    'Create correlation heatmap'
                ],
                'tools': ['df.corr()', 'seaborn.heatmap()'],
                'code_type': 'correlation'
            },
            'visualization': {
                'keywords': ['visualize', 'plot', 'chart', 'graph', 'show', 'draw'],
                'steps': [
                    'Identify variable types',
                    'Select appropriate chart types',
                    'Generate multiple visualizations',
                    'Add labels and titles'
                ],
                'tools': ['matplotlib.pyplot', 'seaborn'],
                'code_type': 'visualization'
            },
            'outlier': {
                'keywords': ['outlier', 'anomaly', 'unusual', 'extreme', 'abnormal'],
                'steps': [
                    'Calculate quartiles (Q1, Q3)',
                    'Compute IQR (Interquartile Range)',
                    'Identify outliers using IQR method',
                    'Visualize with box plots'
                ],
                'tools': ['df.quantile()', 'matplotlib.pyplot.boxplot()'],
                'code_type': 'outlier_detection'
            },
            'predictive': {
                'keywords': ['predict', 'forecast', 'model', 'regression', 'machine learning', 'ml'],
                'steps': [
                    'Identify features and target variable',
                    'Split data into train/test sets',
                    'Train machine learning model',
                    'Evaluate model performance'
                ],
                'tools': ['sklearn.train_test_split', 'sklearn.LinearRegression'],
                'code_type': 'predictive'
            },
            'groupby': {
                'keywords': ['group', 'aggregate', 'segment', 'category', 'breakdown'],
                'steps': [
                    'Identify grouping columns',
                    'Select aggregation functions',
                    'Perform group-by operations',
                    'Visualize grouped results'
                ],
                'tools': ['df.groupby()', 'agg()'],
                'code_type': 'groupby'
            }
        }
    
    def plan(self, query: str, data_info: Dict) -> Dict:
        """Create an analytical plan based on user query"""
        query_lower = query.lower()
        
        # Match query to plan template
        matched_plan = None
        for plan_name, template in self.plan_templates.items():
            if any(keyword in query_lower for keyword in template['keywords']):
                matched_plan = template
                break
        
        # Default to descriptive if no match
        if matched_plan is None:
            matched_plan = self.plan_templates['descriptive']
        
        plan = {
            'query': query,
            'plan_type': matched_plan['code_type'],
            'steps': matched_plan['steps'],
            'tools': matched_plan['tools'],
            'estimated_time': '2-5 seconds',
            'data_context': {
                'rows': data_info.get('rows', 0),
                'columns': data_info.get('columns', 0),
                'numeric_cols': data_info.get('numeric_columns', []),
                'categorical_cols': data_info.get('categorical_columns', [])
            }
        }
        
        return plan


class CodeGeneratorAgent:
    """Agent 2: Generates safe, executable Python code"""
    
    def __init__(self):
        pass
    
    def generate(self, plan: Dict, data_info: Dict) -> Dict:
        """Generate Python code based on the plan"""
        code_type = plan['plan_type']
        
        # Code generation templates
        templates = {
            'descriptive': self._generate_descriptive_code(),
            'correlation': self._generate_correlation_code(),
            'visualization': self._generate_visualization_code(),
            'outlier_detection': self._generate_outlier_code(),
            'predictive': self._generate_predictive_code(),
            'groupby': self._generate_groupby_code()
        }
        
        code = templates.get(code_type, templates['descriptive'])
        
        return {
            'code': code,
            'language': 'python',
            'libraries': ['pandas', 'numpy', 'matplotlib', 'seaborn', 'sklearn'],
            'plan_type': code_type
        }
    
    def _generate_descriptive_code(self):
        return '''# Descriptive Statistics
print("="*60)
print("DESCRIPTIVE STATISTICS")
print("="*60)

stats = df.describe()
print("\\n📊 Statistical Summary:")
print(stats)

missing = df.isnull().sum()
if missing.sum() > 0:
    print("\\n🔍 Missing Values:")
    print(missing[missing > 0])
else:
    print("\\n✅ No missing values")

print(f"\\n📐 Shape: {df.shape[0]} rows × {df.shape[1]} columns")

result = {'statistics': stats.to_dict(), 'shape': df.shape}'''
    
    def _generate_correlation_code(self):
        return '''# Correlation Analysis
print("="*60)
print("CORRELATION ANALYSIS")
print("="*60)

numeric_cols = df.select_dtypes(include=[np.number]).columns
if len(numeric_cols) >= 2:
    corr_matrix = df[numeric_cols].corr()
    
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_pairs.append({
                'feature1': corr_matrix.columns[i],
                'feature2': corr_matrix.columns[j],
                'correlation': corr_matrix.iloc[i, j]
            })
    
    corr_pairs = sorted(corr_pairs, key=lambda x: abs(x['correlation']), reverse=True)
    
    print("\\n🔗 Top 5 Correlations:")
    for idx, pair in enumerate(corr_pairs[:5], 1):
        print(f"{idx}. {pair['feature1']} ↔ {pair['feature2']}: {pair['correlation']:.3f}")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', ax=ax)
    ax.set_title('Correlation Heatmap')
    plt.tight_layout()
    
    result = {'top_correlations': corr_pairs[:10]}
else:
    print("\\n⚠️ Need at least 2 numeric columns")
    result = {'error': 'Insufficient columns'}'''
    
    def _generate_visualization_code(self):
        return '''# Data Visualization
print("="*60)
print("VISUALIZATION")
print("="*60)

numeric_cols = df.select_dtypes(include=['number']).columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Data Exploration Dashboard', fontsize=16, fontweight='bold')

if len(numeric_cols) > 0:
    df[numeric_cols[0]].hist(bins=30, ax=axes[0,0], edgecolor='black', color='skyblue')
    axes[0,0].set_title(f'Distribution: {numeric_cols[0]}')
    axes[0,0].grid(alpha=0.3)

if len(numeric_cols) >= 2:
    df[numeric_cols[:min(5, len(numeric_cols))]].boxplot(ax=axes[0,1])
    axes[0,1].set_title('Box Plot')
    axes[0,1].tick_params(axis='x', rotation=45)

if len(categorical_cols) > 0:
    df[categorical_cols[0]].value_counts().head(10).plot(kind='bar', ax=axes[1,0], color='coral')
    axes[1,0].set_title(f'Top 10: {categorical_cols[0]}')
    axes[1,0].tick_params(axis='x', rotation=45)

if len(numeric_cols) >= 2:
    corr = df[numeric_cols[:min(6, len(numeric_cols))]].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=axes[1,1], center=0, fmt='.2f')
    axes[1,1].set_title('Correlation Heatmap')

plt.tight_layout()
print("\\n✅ Visualizations created")

result = {'message': 'Visualizations complete'}'''
    
    def _generate_outlier_code(self):
        return '''# Outlier Detection
print("="*60)
print("OUTLIER DETECTION")
print("="*60)

numeric_cols = df.select_dtypes(include=[np.number]).columns
outliers_info = {}
total = 0

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    count = len(outliers)
    total += count
    outliers_info[col] = {'count': count, 'percentage': (count/len(df))*100}
    print(f"\\n{col}: {count} outliers ({outliers_info[col]['percentage']:.1f}%)")

print(f"\\n🎯 Total: {total} outliers")

fig, axes = plt.subplots(1, min(4, len(numeric_cols)), figsize=(15, 4))
if len(numeric_cols) == 1:
    axes = [axes]

for idx, col in enumerate(numeric_cols[:4]):
    df.boxplot(column=col, ax=axes[idx])
    axes[idx].set_title(f'{col}\\n{outliers_info[col]["count"]} outliers')

plt.tight_layout()

result = {'outliers_info': outliers_info, 'total_outliers': total}'''
    
    def _generate_predictive_code(self):
        return '''# Predictive Modeling
print("="*60)
print("PREDICTIVE MODEL")
print("="*60)

numeric_cols = df.select_dtypes(include=[np.number]).columns

if len(numeric_cols) >= 2:
    X = df[numeric_cols[:-1]].fillna(df[numeric_cols[:-1]].mean())
    y = df[numeric_cols[-1]].fillna(df[numeric_cols[-1]].mean())
    
    print(f"\\nFeatures: {', '.join(numeric_cols[:-1])}")
    print(f"Target: {numeric_cols[-1]}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"\\n✅ R² Score: {r2:.4f}")
    print(f"✅ RMSE: {rmse:.4f}")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, alpha=0.6)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title(f'Predictions (R² = {r2:.3f})')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    
    result = {'r2_score': r2, 'rmse': rmse}
else:
    print("\\n⚠️ Need at least 2 numeric columns")
    result = {'error': 'Insufficient columns'}'''
    
    def _generate_groupby_code(self):
        return '''# Group-By Analysis
print("="*60)
print("GROUP-BY ANALYSIS")
print("="*60)

cat_cols = df.select_dtypes(include=['object', 'category']).columns
num_cols = df.select_dtypes(include=[np.number]).columns

if len(cat_cols) > 0 and len(num_cols) > 0:
    group_col = cat_cols[0]
    agg_col = num_cols[0]
    
    print(f"\\nGrouping by: {group_col}")
    print(f"Aggregating: {agg_col}")
    
    grouped = df.groupby(group_col)[agg_col].agg(['mean', 'count'])
    print(f"\\n{grouped}")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    grouped['mean'].plot(kind='bar', ax=axes[0], color='skyblue')
    axes[0].set_title(f'Average {agg_col}')
    axes[0].tick_params(axis='x', rotation=45)
    
    grouped['count'].plot(kind='bar', ax=axes[1], color='coral')
    axes[1].set_title('Count')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    result = {'grouped': grouped.to_dict()}
else:
    print("\\n⚠️ Need categorical and numeric columns")
    result = {'error': 'Insufficient column types'}'''


class VerifierAgent:
    """Agent 3: Verifies code safety before execution"""
    
    def __init__(self):
        self.dangerous_patterns = [
            'import os', 'import sys', 'import subprocess',
            '__import__', 'eval(', 'exec(', 'compile(',
            'open(', 'file(', 'input(', 'requests.',
            'urllib.', 'socket.', 'pickle.', 'rm -rf'
        ]
        
        self.allowed_imports = [
            'pandas', 'numpy', 'matplotlib', 'seaborn',
            'sklearn', 'scipy', 'statsmodels'
        ]
    
    def verify(self, code: str) -> Dict:
        """Verify code is safe to execute"""
        issues = []
        
        # Check for dangerous patterns
        for pattern in self.dangerous_patterns:
            if pattern in code:
                issues.append(f"Dangerous: {pattern}")
        
        # Verify imports using AST
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if not any(alias.name.startswith(lib) for lib in self.allowed_imports):
                            issues.append(f"Unapproved: {alias.name}")
        except SyntaxError as e:
            issues.append(f"Syntax error: {str(e)}")
        
        return {
            'safe': len(issues) == 0,
            'issues': issues,
            'timestamp': datetime.now().isoformat()
        }


class ExecutionAgent:
    """Agent 4: Executes verified code safely"""
    
    def execute(self, code: str, df: pd.DataFrame) -> Dict:
        """Execute code with sandboxed environment"""
        # Safe globals
        safe_globals = {
            '__builtins__': {
                'print': print, 'len': len, 'range': range,
                'enumerate': enumerate, 'zip': zip, 'sorted': sorted,
                'sum': sum, 'min': min, 'max': max, 'abs': abs,
                'round': round, 'str': str, 'int': int, 'float': float,
                'list': list, 'dict': dict, 'tuple': tuple
            },
            'pd': pd,
            'np': np,
            'plt': plt,
            'sns': sns,
            'df': df.copy(),
            'train_test_split': train_test_split,
            'LinearRegression': LinearRegression,
            'mean_squared_error': mean_squared_error,
            'r2_score': r2_score,
            'mean_absolute_error': mean_absolute_error
        }
        
        # Capture output
        old_stdout = sys.stdout
        sys.stdout = captured = io.StringIO()
        
        result = {}
        figure = None
        
        try:
            exec(code, safe_globals)
            if 'result' in safe_globals:
                result = safe_globals['result']
            if plt.get_fignums():
                figure = plt.gcf()
            success = True
            error = None
        except Exception as e:
            success = False
            error = f"{type(e).__name__}: {str(e)}"
            result = {'error': error}
        finally:
            output = captured.getvalue()
            sys.stdout = old_stdout
        
        return {
            'success': success,
            'result': result,
            'output': output,
            'error': error,
            'figure': figure,
            'timestamp': datetime.now().isoformat()
        }


class ExplainerAgent:
    """Agent 5: Explains results in natural language"""
    
    def explain(self, execution_result: Dict, plan: Dict) -> Dict:
        """Generate explanation of results"""
        plan_type = plan['plan_type']
        result = execution_result.get('result', {})
        
        explanations = {
            'descriptive': self._explain_descriptive(result),
            'correlation': self._explain_correlation(result),
            'visualization': self._explain_visualization(result),
            'outlier_detection': self._explain_outliers(result),
            'predictive': self._explain_predictive(result),
            'groupby': self._explain_groupby(result)
        }
        
        explanation = explanations.get(plan_type, "Analysis completed successfully.")
        
        return {
            'explanation': explanation,
            'plan_type': plan_type,
            'success': execution_result['success']
        }
    
    def _explain_descriptive(self, result):
        text = "📊 **Descriptive Analysis Complete**\n\n"
        if 'shape' in result:
            rows, cols = result['shape']
            text += f"**Dataset Size:** {rows:,} rows × {cols} columns\n\n"
        text += "I calculated comprehensive statistics including mean, median, std, min, and max for all numeric columns.\n\n"
        text += "**Key Actions:**\n"
        text += "• Computed descriptive statistics\n"
        text += "• Checked for missing values\n"
        text += "• Analyzed data types\n"
        return text
    
    def _explain_correlation(self, result):
        text = "🔗 **Correlation Analysis Complete**\n\n"
        text += "I examined relationships between numeric variables using correlation analysis.\n\n"
        if 'top_correlations' in result:
            text += "**Top Correlations:**\n"
            for i, pair in enumerate(result['top_correlations'][:5], 1):
                corr = pair['correlation']
                strength = 'Strong' if abs(corr) > 0.7 else 'Moderate' if abs(corr) > 0.4 else 'Weak'
                text += f"{i}. {pair['feature1']} ↔ {pair['feature2']}: {corr:.3f} ({strength})\n"
            text += "\n**Interpretation:** Correlations near ±1 indicate strong linear relationships.\n"
        return text
    
    def _explain_visualization(self, result):
        text = "📈 **Visual Analysis Complete**\n\n"
        text += "I created a comprehensive dashboard with multiple visualizations:\n\n"
        text += "**Charts:**\n"
        text += "• Distribution plot for data spread\n"
        text += "• Box plot showing outliers\n"
        text += "• Bar chart for categories\n"
        text += "• Correlation heatmap\n"
        return text
    
    def _explain_outliers(self, result):
        text = "🎯 **Outlier Detection Complete**\n\n"
        text += "I used the IQR method to identify outliers:\n\n"
        if 'total_outliers' in result:
            text += f"**Total Outliers:** {result['total_outliers']}\n\n"
        if 'outliers_info' in result:
            text += "**By Column:**\n"
            for col, info in result['outliers_info'].items():
                status = "⚠️" if info['count'] > 0 else "✅"
                text += f"• {col}: {info['count']} outliers ({info['percentage']:.1f}%) {status}\n"
        return text
    
    def _explain_predictive(self, result):
        text = "🤖 **Predictive Model Complete**\n\n"
        if 'r2_score' in result:
            r2 = result['r2_score']
            text += f"**R² Score:** {r2:.4f}\n"
            text += f"**RMSE:** {result.get('rmse', 0):.4f}\n\n"
            performance = 'Excellent' if r2 > 0.8 else 'Good' if r2 > 0.6 else 'Moderate'
            text += f"**Performance:** {performance} - Model explains {r2*100:.1f}% of variance\n"
        return text
    
    def _explain_groupby(self, result):
        text = "📊 **Group-By Analysis Complete**\n\n"
        text += "I performed aggregation analysis by category:\n\n"
        text += "**Actions:**\n"
        text += "• Grouped data by categorical variable\n"
        text += "• Calculated mean and count for each group\n"
        text += "• Created comparison visualizations\n"
        return text