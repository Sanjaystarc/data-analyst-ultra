"""
core_agent.py — TRUE Agentic AI
LangChain Agent + Tools + Memory + Gemini
"""

import os
import json
import warnings
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.agent import AgentExecutor
from langchain.agents.tool_calling_agent.base import create_tool_calling_agent
from langchain_community.chat_message_histories import ChatMessageHistory

warnings.filterwarnings("ignore")
load_dotenv()

PALETTE = ["#6C63FF", "#FF6584", "#43E97B", "#F7971E", "#4FC3F7", "#CE93D8"]
DARK_BG  = "#0F0F1A"
CARD_BG  = "#1A1A2E"

_df: pd.DataFrame = None
_profile: dict = None

def set_dataframe(df, profile):
    global _df, _profile
    _df = df
    _profile = profile

def get_llm(api_key: str):
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.3,
        convert_system_message_to_human=True,
    )

def load_file(file):
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file), "CSV"
    elif name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file), "Excel"
    elif name.endswith(".json"):
        content = json.load(file)
        if isinstance(content, list):
            df = pd.DataFrame(content)
        else:
            df = pd.DataFrame(content) if any(isinstance(v, list) for v in content.values()) else pd.DataFrame([content])
        return df, "JSON"
    else:
        raise ValueError(f"Unsupported file type: {name}")

def profile_dataframe(df):
    numeric_cols  = df.select_dtypes(include="number").columns.tolist()
    category_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime"]).columns.tolist()
    profile = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "numeric_columns": numeric_cols,
        "categorical_columns": category_cols,
        "datetime_columns": datetime_cols,
        "null_counts": df.isnull().sum().to_dict(),
        "null_pct": (df.isnull().mean() * 100).round(2).to_dict(),
        "duplicates": int(df.duplicated().sum()),
    }
    if numeric_cols:
        profile["numeric_stats"] = df[numeric_cols].describe().round(3).to_dict()
    if category_cols:
        profile["top_categories"] = {col: df[col].value_counts().head(5).to_dict() for col in category_cols}
    return profile

def profile_to_text(profile, df):
    rows, cols = profile["shape"]
    lines = [
        f"Dataset: {rows} rows x {cols} columns",
        f"Numeric columns : {', '.join(profile['numeric_columns']) or 'None'}",
        f"Categorical cols : {', '.join(profile['categorical_columns']) or 'None'}",
        f"Datetime cols    : {', '.join(profile['datetime_columns']) or 'None'}",
        f"Missing values   : {sum(profile['null_counts'].values())} total",
        f"Duplicate rows   : {profile['duplicates']}",
        "", "--- Sample Data (first 5 rows) ---",
        df.head(5).to_string(index=False),
    ]
    if profile.get("numeric_stats"):
        lines += ["", "--- Numeric Stats ---"]
        for col, stats in profile["numeric_stats"].items():
            lines.append(f"  {col}: mean={stats.get('mean','?')}, std={stats.get('std','?')}, min={stats.get('min','?')}, max={stats.get('max','?')}")
    return "\n".join(lines)

# ══════════════════════════════════════════════
# AGENT TOOLS
# ══════════════════════════════════════════════

@tool
def profile_data(query: str) -> str:
    """Get full statistical profile of the dataset. Use this FIRST before any analysis."""
    if _df is None:
        return "No dataset loaded. Please upload a file first."
    return profile_to_text(_profile, _df)

@tool
def analyze_column(column_name: str) -> str:
    """Deeply analyze a specific column. Provide the exact column name."""
    if _df is None:
        return "No dataset loaded."
    if column_name not in _df.columns:
        return f"Column '{column_name}' not found. Available: {_df.columns.tolist()}"
    col = _df[column_name]
    result = [f"Analysis of '{column_name}'", f"Type: {col.dtype}",
              f"Non-null: {col.count()} / {len(col)}", f"Nulls: {col.isnull().sum()} ({col.isnull().mean()*100:.1f}%)"]
    if pd.api.types.is_numeric_dtype(col):
        Q1, Q3 = col.quantile(0.25), col.quantile(0.75)
        IQR = Q3 - Q1
        outliers = int(((col < Q1 - 1.5*IQR) | (col > Q3 + 1.5*IQR)).sum())
        result += [f"Mean: {col.mean():.3f}", f"Median: {col.median():.3f}",
                   f"Std: {col.std():.3f}", f"Min: {col.min()}", f"Max: {col.max()}",
                   f"Skewness: {col.skew():.3f}", f"Outliers: {outliers}"]
    else:
        result += [f"Unique values: {col.nunique()}",
                   f"Top 5: {col.value_counts().head(5).to_dict()}",
                   f"Most common: {col.mode()[0] if not col.mode().empty else 'N/A'}"]
    return "\n".join(result)

@tool
def find_correlations(query: str) -> str:
    """Find correlations between numeric columns. Highlights strong relationships."""
    if _df is None:
        return "No dataset loaded."
    num_cols = _profile["numeric_columns"]
    if len(num_cols) < 2:
        return "Need at least 2 numeric columns."
    corr = _df[num_cols].corr().round(3)
    strong = []
    for i in range(len(num_cols)):
        for j in range(i+1, len(num_cols)):
            val = corr.iloc[i, j]
            if abs(val) >= 0.5:
                strength = "strong" if abs(val) >= 0.8 else "moderate"
                direction = "positive" if val > 0 else "negative"
                strong.append(f"  {num_cols[i]} <-> {num_cols[j]}: {val} ({strength} {direction})")
    result = ["Correlation Matrix:", corr.to_string()]
    if strong:
        result += ["", "Notable correlations:"] + strong
    else:
        result.append("No strong correlations found (|r| >= 0.5)")
    return "\n".join(result)

@tool
def detect_anomalies(query: str) -> str:
    """Detect outliers and anomalies across all numeric columns using IQR method."""
    if _df is None:
        return "No dataset loaded."
    num_cols = _profile["numeric_columns"]
    if not num_cols:
        return "No numeric columns found."
    results = ["Anomaly Detection Report:"]
    total = 0
    for col in num_cols:
        series = _df[col].dropna()
        Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
        IQR = Q3 - Q1
        outliers = _df[((_df[col] < Q1 - 1.5*IQR) | (_df[col] > Q3 + 1.5*IQR))][col]
        if len(outliers) > 0:
            total += len(outliers)
            results.append(f"  {col}: {len(outliers)} outliers | Examples: {outliers.head(3).tolist()}")
    results.append(f"\nTotal outliers: {total}")
    if total == 0:
        results.append("No significant outliers detected.")
    return "\n".join(results)

@tool
def run_aggregation(query: str) -> str:
    """
    Compute group-by aggregations.
    Format input as: 'group_col|agg_col|function'
    Example: 'category|sales|sum'
    Supported: sum, mean, count, max, min, median
    """
    if _df is None:
        return "No dataset loaded."
    try:
        parts = [p.strip() for p in query.split("|")]
        if len(parts) == 3:
            group_col, agg_col, func = parts
        elif len(parts) == 2:
            group_col, agg_col, func = parts[0], parts[1], "mean"
        else:
            cat_cols = _profile["categorical_columns"]
            num_cols = _profile["numeric_columns"]
            if not cat_cols or not num_cols:
                return "Could not determine columns."
            group_col, agg_col, func = cat_cols[0], num_cols[0], "sum"
        if group_col not in _df.columns:
            return f"Column '{group_col}' not found. Available: {_df.columns.tolist()}"
        if agg_col not in _df.columns:
            return f"Column '{agg_col}' not found. Available: {_df.columns.tolist()}"
        fn = func.lower()
        result = _df.groupby(group_col)[agg_col].agg(fn).reset_index().sort_values(agg_col, ascending=False)
        result.columns = [group_col, f"{fn}_{agg_col}"]
        return f"Aggregation: {fn.upper()} of '{agg_col}' by '{group_col}'\n{result.to_string(index=False)}"
    except Exception as e:
        return f"Aggregation error: {str(e)}"

@tool
def generate_insight_report(query: str) -> str:
    """Generate a complete automated insight report with data quality score, patterns, and recommendations."""
    if _df is None:
        return "No dataset loaded."
    rows, cols = _profile["shape"]
    num_cols = _profile["numeric_columns"]
    cat_cols = _profile["categorical_columns"]
    nulls = sum(_profile["null_counts"].values())
    null_pct = (nulls / (rows * cols) * 100) if rows * cols > 0 else 0
    quality = 100
    if null_pct > 20: quality -= 30
    elif null_pct > 10: quality -= 15
    elif null_pct > 5: quality -= 5
    if _profile["duplicates"] > 0: quality -= 10
    report = [
        "=" * 50, "AUTOMATED INSIGHT REPORT", "=" * 50, "",
        "1. DATASET OVERVIEW",
        f"   Rows: {rows:,} | Columns: {cols}",
        f"   Numeric: {len(num_cols)} | Categorical: {len(cat_cols)}",
        f"   Data Quality Score: {quality}/100", "",
        "2. DATA QUALITY",
        f"   Missing values: {nulls} ({null_pct:.1f}%)",
        f"   Duplicate rows: {_profile['duplicates']}",
    ]
    if nulls > 0:
        worst = max(_profile["null_pct"].items(), key=lambda x: x[1])
        report.append(f"   Worst column: '{worst[0]}' ({worst[1]}% missing)")
    report += ["", "3. KEY STATISTICS"]
    for col in num_cols[:5]:
        stats = _profile.get("numeric_stats", {}).get(col, {})
        report.append(f"   {col}: mean={stats.get('mean','?')}, range=[{stats.get('min','?')}, {stats.get('max','?')}]")
    if cat_cols:
        report += ["", "4. CATEGORICAL SUMMARY"]
        for col in cat_cols[:3]:
            top = _df[col].value_counts().index[0] if not _df[col].empty else "N/A"
            report.append(f"   {col}: {_df[col].nunique()} unique | most common = '{top}'")
    report += [
        "", "5. RECOMMENDATIONS",
        f"   - {'Fix missing values' if null_pct > 5 else 'Data completeness looks good'}",
        f"   - {'Remove duplicate rows' if _profile['duplicates'] > 0 else 'No duplicates found'}",
        f"   - {'Run correlation analysis' if len(num_cols) >= 2 else 'Need more numeric columns'}",
        f"   - {'Encode categorical columns for ML' if cat_cols else 'Add categorical features'}",
        "", "=" * 50,
    ]
    return "\n".join(report)

@tool
def recommend_chart(question: str) -> str:
    """Recommend best chart type for a question. Returns JSON with chart_type, x_col, y_col."""
    if _profile is None:
        return json.dumps({"chart_type": "bar_chart", "x_col": None, "y_col": None})
    num_cols = _profile["numeric_columns"]
    cat_cols = _profile["categorical_columns"]
    dt_cols  = _profile["datetime_columns"]
    q = question.lower()
    if any(w in q for w in ["trend", "over time", "time", "date"]) and dt_cols and num_cols:
        return json.dumps({"chart_type": "time_series", "x_col": dt_cols[0], "y_col": num_cols[0]})
    elif any(w in q for w in ["correlat", "relationship", "vs", "versus"]) and len(num_cols) >= 2:
        return json.dumps({"chart_type": "correlation_heatmap", "x_col": None, "y_col": None})
    elif any(w in q for w in ["distribut", "spread", "histogram"]) and num_cols:
        return json.dumps({"chart_type": "distribution_plots", "x_col": None, "y_col": num_cols[0]})
    elif any(w in q for w in ["outlier", "box", "range"]) and num_cols:
        return json.dumps({"chart_type": "box_plots", "x_col": None, "y_col": None})
    elif any(w in q for w in ["proportion", "share", "percent", "pie"]) and cat_cols:
        return json.dumps({"chart_type": "pie_chart", "x_col": cat_cols[0], "y_col": None})
    elif cat_cols and num_cols:
        return json.dumps({"chart_type": "bar_chart", "x_col": cat_cols[0], "y_col": num_cols[0]})
    elif len(num_cols) >= 2:
        return json.dumps({"chart_type": "scatter", "x_col": num_cols[0], "y_col": num_cols[1]})
    return json.dumps({"chart_type": "bar_chart", "x_col": None, "y_col": None})

# ══════════════════════════════════════════════
# AGENT BUILDER
# ══════════════════════════════════════════════

TOOLS = [profile_data, analyze_column, find_correlations,
         detect_anomalies, run_aggregation, generate_insight_report, recommend_chart]

SYSTEM_PROMPT = """You are DataMind, an expert autonomous data analyst AI agent.

You have access to powerful tools to analyze any dataset. When a user asks a question:
1. THINK about what tools you need
2. PLAN your steps (use multiple tools in sequence when needed)
3. EXECUTE each tool
4. SYNTHESIZE the results into a clear, insightful answer
5. SELF-CORRECT if a tool returns an error

Your tools:
- profile_data: Get dataset overview (use this first)
- analyze_column: Deep dive into a specific column
- find_correlations: Find relationships between numeric columns
- detect_anomalies: Find outliers and data quality issues
- run_aggregation: Group-by calculations
- generate_insight_report: Full automated analysis report
- recommend_chart: Suggest best visualization

Always be precise, proactive, and thorough. Use multiple tools when needed.
Remember conversation history and refer to previous questions when relevant."""

def build_agent(llm) -> AgentExecutor:
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, TOOLS, prompt)
    return AgentExecutor(
        agent=agent, tools=TOOLS, verbose=True,
        max_iterations=6, early_stopping_method="generate",
        handle_parsing_errors=True, return_intermediate_steps=True,
    )

def run_agent(question: str, agent_executor: AgentExecutor, chat_history: list) -> dict:
    try:
        result = agent_executor.invoke({"input": question, "chat_history": chat_history})
        return {"output": result.get("output", "No response."), "steps": result.get("intermediate_steps", []), "error": None}
    except Exception as e:
        return {"output": f"Agent error: {str(e)}", "steps": [], "error": str(e)}

# ── Chart Engine ──────────────────────────────
def auto_suggest_charts(profile):
    suggestions = []
    if len(profile["numeric_columns"]) >= 2:
        suggestions.extend(["correlation_heatmap", "scatter_matrix"])
    if profile["numeric_columns"]:
        suggestions.extend(["distribution_plots", "box_plots"])
    if profile["categorical_columns"] and profile["numeric_columns"]:
        suggestions.extend(["bar_chart", "pie_chart"])
    if profile["datetime_columns"] and profile["numeric_columns"]:
        suggestions.append("time_series")
    return suggestions

def make_plotly_chart(chart_type, df, profile, x_col=None, y_col=None, color_col=None):
    num_cols = profile["numeric_columns"]
    cat_cols = profile["categorical_columns"]
    template = "plotly_dark"
    if chart_type == "correlation_heatmap" and len(num_cols) >= 2:
        fig = px.imshow(df[num_cols].corr().round(2), text_auto=True,
                        color_continuous_scale="RdBu_r", title="Correlation Heatmap",
                        template=template, color_continuous_midpoint=0)
    elif chart_type == "distribution_plots" and num_cols:
        col = y_col or num_cols[0]
        fig = px.histogram(df, x=col, nbins=30, marginal="box",
                           title=f"Distribution of {col}",
                           color_discrete_sequence=PALETTE, template=template)
    elif chart_type == "box_plots" and num_cols:
        fig = go.Figure()
        for i, col in enumerate(num_cols[:6]):
            fig.add_trace(go.Box(y=df[col], name=col, marker_color=PALETTE[i % len(PALETTE)]))
        fig.update_layout(title="Box Plots", template=template)
    elif chart_type == "bar_chart" and cat_cols and num_cols:
        xc, yc = x_col or cat_cols[0], y_col or num_cols[0]
        agg = df.groupby(xc)[yc].mean().reset_index().sort_values(yc, ascending=False).head(15)
        fig = px.bar(agg, x=xc, y=yc, color=yc, color_continuous_scale="Viridis",
                     title=f"Average {yc} by {xc}", template=template)
    elif chart_type == "pie_chart" and cat_cols:
        col = x_col or cat_cols[0]
        counts = df[col].value_counts().head(8)
        fig = px.pie(values=counts.values, names=counts.index,
                     title=f"Distribution of {col}",
                     color_discrete_sequence=PALETTE, template=template)
    elif chart_type == "scatter_matrix" and len(num_cols) >= 2:
        fig = px.scatter_matrix(df, dimensions=num_cols[:4],
                                color=cat_cols[0] if cat_cols else None,
                                color_discrete_sequence=PALETTE, title="Scatter Matrix", template=template)
        fig.update_traces(diagonal_visible=False, showupperhalf=False)
    elif chart_type == "time_series" and profile["datetime_columns"] and num_cols:
        dt_col = profile["datetime_columns"][0]
        yc = y_col or num_cols[0]
        fig = px.line(df.sort_values(dt_col), x=dt_col, y=yc,
                      title=f"{yc} over Time", color_discrete_sequence=PALETTE, template=template)
    elif chart_type == "scatter" and len(num_cols) >= 2:
        xc, yc = x_col or num_cols[0], y_col or num_cols[1]
        fig = px.scatter(df, x=xc, y=yc,
                         color=color_col or (cat_cols[0] if cat_cols else None),
                         color_discrete_sequence=PALETTE, title=f"{xc} vs {yc}",
                         trendline="ols", template=template)
    elif chart_type == "line" and num_cols:
        xc = x_col or (profile["datetime_columns"][0] if profile["datetime_columns"] else num_cols[0])
        yc = y_col or num_cols[0]
        fig = px.line(df, x=xc, y=yc, color_discrete_sequence=PALETTE,
                      title=f"{yc} trend", template=template)
    else:
        if num_cols:
            means = df[num_cols[:8]].mean()
            fig = px.bar(x=means.index, y=means.values, color=means.values,
                         color_continuous_scale="Viridis", title="Column Means", template=template)
        else:
            fig = go.Figure()
            fig.update_layout(template=template, title="Chart Unavailable")
    fig.update_layout(paper_bgcolor=DARK_BG, plot_bgcolor=CARD_BG,
                      font=dict(family="DM Sans, sans-serif", color="#E0E0FF"),
                      margin=dict(l=40, r=40, t=60, b=40))
    return fig
