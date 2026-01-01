"""
Utility Functions for Data Loading and Processing
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import io


def load_data(uploaded_file) -> Optional[pd.DataFrame]:
    """
    Load CSV or Excel file into a pandas DataFrame
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        pd.DataFrame or None if loading fails
    """
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'csv':
            # Try different encodings
            try:
                df = pd.read_csv(uploaded_file)
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding='latin-1')
        
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file)
        
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        # Basic validation
        if df.empty:
            raise ValueError("The uploaded file is empty")
        
        return df
    
    except Exception as e:
        print(f"Error loading file: {str(e)}")
        return None


def get_data_info(df: pd.DataFrame) -> Dict:
    """
    Extract metadata and summary information from DataFrame
    
    Args:
        df: pandas DataFrame
        
    Returns:
        Dictionary containing data information
    """
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Infer datetime columns
    datetime_columns = []
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            try:
                pd.to_datetime(df[col])
                datetime_columns.append(col)
            except:
                pass
    
    info = {
        'rows': len(df),
        'columns': len(df.columns),
        'column_names': df.columns.tolist(),
        'column_types': df.dtypes.astype(str).to_dict(),
        'numeric_columns': numeric_columns,
        'categorical_columns': categorical_columns,
        'datetime_columns': datetime_columns,
        'missing_values': df.isnull().sum().to_dict(),
        'total_missing': df.isnull().sum().sum(),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
    }
    
    # Add sample statistics for numeric columns
    if numeric_columns:
        info['numeric_summary'] = {
            col: {
                'min': df[col].min(),
                'max': df[col].max(),
                'mean': df[col].mean(),
                'median': df[col].median()
            }
            for col in numeric_columns
        }
    
    return info


def validate_dataframe(df: pd.DataFrame, max_rows: int = 1000000) -> tuple:
    """
    Validate DataFrame for processing
    
    Args:
        df: pandas DataFrame
        max_rows: Maximum allowed rows
        
    Returns:
        Tuple of (is_valid: bool, message: str)
    """
    if df is None:
        return False, "DataFrame is None"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) > max_rows:
        return False, f"DataFrame has {len(df)} rows, exceeds maximum of {max_rows}"
    
    if len(df.columns) == 0:
        return False, "DataFrame has no columns"
    
    # Check for extremely large memory usage (> 500MB)
    memory_mb = df.memory_usage(deep=True).sum() / 1024**2
    if memory_mb > 500:
        return False, f"DataFrame uses {memory_mb:.1f}MB, exceeds 500MB limit"
    
    return True, "DataFrame is valid"


def sample_dataframe(df: pd.DataFrame, max_rows: int = 10000) -> pd.DataFrame:
    """
    Sample large DataFrame for faster processing
    
    Args:
        df: pandas DataFrame
        max_rows: Maximum rows to keep
        
    Returns:
        Sampled DataFrame
    """
    if len(df) <= max_rows:
        return df
    
    # Stratified sampling if possible
    if len(df.select_dtypes(include=['object', 'category']).columns) > 0:
        try:
            # Sample proportionally from first categorical column
            cat_col = df.select_dtypes(include=['object', 'category']).columns[0]
            return df.groupby(cat_col, group_keys=False).apply(
                lambda x: x.sample(min(len(x), max_rows // df[cat_col].nunique()))
            )
        except:
            pass
    
    # Random sampling as fallback
    return df.sample(n=max_rows, random_state=42)


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean DataFrame column names for easier processing
    
    Args:
        df: pandas DataFrame
        
    Returns:
        DataFrame with cleaned column names
    """
    df = df.copy()
    
    # Remove leading/trailing whitespace
    df.columns = df.columns.str.strip()
    
    # Replace spaces and special characters with underscores
    df.columns = df.columns.str.replace(r'[^\w]', '_', regex=True)
    
    # Remove consecutive underscores
    df.columns = df.columns.str.replace(r'_+', '_', regex=True)
    
    # Remove trailing underscores
    df.columns = df.columns.str.rstrip('_')
    
    # Convert to lowercase
    df.columns = df.columns.str.lower()
    
    return df


def detect_and_convert_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Automatically detect and convert column data types
    
    Args:
        df: pandas DataFrame
        
    Returns:
        DataFrame with optimized data types
    """
    df = df.copy()
    
    for col in df.columns:
        # Skip if already numeric or datetime
        if df[col].dtype in ['int64', 'float64', 'datetime64[ns]']:
            continue
        
        # Try converting to numeric
        try:
            df[col] = pd.to_numeric(df[col])
            continue
        except:
            pass
        
        # Try converting to datetime
        if 'date' in col.lower() or 'time' in col.lower():
            try:
                df[col] = pd.to_datetime(df[col])
                continue
            except:
                pass
        
        # Convert to category if low cardinality
        if df[col].dtype == 'object':
            n_unique = df[col].nunique()
            if n_unique < len(df) * 0.5:  # Less than 50% unique values
                df[col] = df[col].astype('category')
    
    return df


def get_column_info(df: pd.DataFrame, column: str) -> Dict:
    """
    Get detailed information about a specific column
    
    Args:
        df: pandas DataFrame
        column: Column name
        
    Returns:
        Dictionary with column information
    """
    if column not in df.columns:
        return {'error': f'Column {column} not found'}
    
    col_data = df[column]
    
    info = {
        'name': column,
        'dtype': str(col_data.dtype),
        'count': len(col_data),
        'null_count': col_data.isnull().sum(),
        'null_percentage': (col_data.isnull().sum() / len(col_data)) * 100,
        'unique_count': col_data.nunique()
    }
    
    # Add type-specific information
    if np.issubdtype(col_data.dtype, np.number):
        info.update({
            'min': col_data.min(),
            'max': col_data.max(),
            'mean': col_data.mean(),
            'median': col_data.median(),
            'std': col_data.std(),
            'q25': col_data.quantile(0.25),
            'q75': col_data.quantile(0.75)
        })
    
    elif col_data.dtype == 'object' or col_data.dtype.name == 'category':
        value_counts = col_data.value_counts()
        info.update({
            'top_values': value_counts.head(10).to_dict(),
            'mode': col_data.mode()[0] if len(col_data.mode()) > 0 else None
        })
    
    return info


def format_number(num: float, decimals: int = 2) -> str:
    """
    Format number for display
    
    Args:
        num: Number to format
        decimals: Number of decimal places
        
    Returns:
        Formatted string
    """
    if abs(num) >= 1e6:
        return f"{num/1e6:.{decimals}f}M"
    elif abs(num) >= 1e3:
        return f"{num/1e3:.{decimals}f}K"
    else:
        return f"{num:.{decimals}f}"


def generate_data_report(df: pd.DataFrame) -> str:
    """
    Generate a text summary report of the DataFrame
    
    Args:
        df: pandas DataFrame
        
    Returns:
        Formatted report string
    """
    info = get_data_info(df)
    
    report = "=" * 60 + "\n"
    report += "DATA SUMMARY REPORT\n"
    report += "=" * 60 + "\n\n"
    
    report += f"Dataset Shape: {info['rows']:,} rows × {info['columns']} columns\n"
    report += f"Memory Usage: {info['memory_usage']:.2f} MB\n"
    report += f"Total Missing Values: {info['total_missing']:,}\n\n"
    
    report += "Column Types:\n"
    report += f"  Numeric: {len(info['numeric_columns'])}\n"
    report += f"  Categorical: {len(info['categorical_columns'])}\n"
    report += f"  Datetime: {len(info['datetime_columns'])}\n\n"
    
    if info['numeric_columns']:
        report += "Numeric Columns:\n"
        for col in info['numeric_columns'][:10]:  # First 10
            report += f"  - {col}\n"
        if len(info['numeric_columns']) > 10:
            report += f"  ... and {len(info['numeric_columns']) - 10} more\n"
        report += "\n"
    
    if info['categorical_columns']:
        report += "Categorical Columns:\n"
        for col in info['categorical_columns'][:10]:  # First 10
            report += f"  - {col}\n"
        if len(info['categorical_columns']) > 10:
            report += f"  ... and {len(info['categorical_columns']) - 10} more\n"
        report += "\n"
    
    # Missing values summary
    missing_cols = {k: v for k, v in info['missing_values'].items() if v > 0}
    if missing_cols:
        report += "Columns with Missing Values:\n"
        for col, count in sorted(missing_cols.items(), key=lambda x: x[1], reverse=True)[:10]:
            pct = (count / info['rows']) * 100
            report += f"  - {col}: {count:,} ({pct:.1f}%)\n"
    
    report += "\n" + "=" * 60 + "\n"
    
    return report