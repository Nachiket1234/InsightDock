import os
import time
import hashlib
import io
from datetime import datetime
import streamlit as st
import pandas as pd
import plotly.express as px
from app_secrets import load_tokens_from_file
from data_loader import download_olist_dataset, load_into_duckdb, describe_schema
from sql_agent import answer_question
from memory import ConversationMemory
from rag import build_schema_docs, build_rag_index, query_index
from hybrid_llm import get_available_providers


st.set_page_config(page_title="Brazil Customer Analytics", layout="wide")
load_tokens_from_file()

# Custom CSS for professional styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}
.hero-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 10px;
    margin-bottom: 2rem;
    text-align: center;
    color: white;
}
.hero-header h1 {
    margin: 0;
    font-size: 2.5rem;
    font-weight: 700;
}
.hero-header p {
    margin: 0.5rem 0 0 0;
    font-size: 1.1rem;
    opacity: 0.9;
}
.metric-card {
    background: rgba(255, 255, 255, 0.1);
    padding: 1rem;
    border-radius: 8px;
    text-align: center;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
}
.metric-card .metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
}
.metric-card .metric-label {
    font-size: 0.9rem;
    opacity: 0.8;
}
</style>
""", unsafe_allow_html=True)

# Professional Header
st.markdown("""
<div class="hero-header">
    <h1>Brazil Customer Analytics</h1>
    <p>Geospatial Dashboard for Olist E-commerce Data</p>
</div>
""", unsafe_allow_html=True)

# Key Metrics Overview
if st.session_state.get("conn"):
    try:
        # Calculate key metrics
        total_customers = st.session_state.conn.execute("SELECT COUNT(DISTINCT customer_unique_id) FROM olist_customers_dataset").fetchone()[0]
        total_revenue = st.session_state.conn.execute("SELECT SUM(price) FROM olist_order_items_dataset").fetchone()[0] or 0
        avg_rating = st.session_state.conn.execute("SELECT AVG(review_score) FROM olist_order_reviews_dataset").fetchone()[0] or 0
        
        # Top states by orders
        top_states = st.session_state.conn.execute("""
            SELECT c.customer_state, COUNT(*) as orders
            FROM olist_customers_dataset c
            JOIN olist_orders_dataset o ON c.customer_id = o.customer_id
            GROUP BY c.customer_state
            ORDER BY orders DESC
            LIMIT 3
        """).fetchall()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{total_customers:,}</div>
                <div class="metric-label">Total Customers</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">${total_revenue:,.0f}</div>
                <div class="metric-label">Total Revenue</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{avg_rating:.1f} ⭐</div>
                <div class="metric-label">Average Rating</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            top_state_text = ", ".join([f"{state[0]}" for state in top_states])
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{top_state_text}</div>
                <div class="metric-label">Top States</div>
            </div>
            """, unsafe_allow_html=True)
    except Exception:
        pass
else:
    st.info("Load dataset to view key metrics")

st.sidebar.title("Data")
st.sidebar.write("Kaggle config: ensure kaggle.json is present in project root.")
if st.sidebar.button("Download Olist dataset"):
    with st.spinner("Downloading dataset from Kaggle..."):
        p = download_olist_dataset()
        st.success(f"Downloaded to {p}")

if "conn" not in st.session_state:
    if (os.path.isdir("data/olist")):
        st.session_state.conn = load_into_duckdb()
        st.session_state.schema = describe_schema(st.session_state.conn)
    else:
        st.session_state.conn = None
        st.session_state.schema = ""

if st.sidebar.button("Load into DuckDB"):
    try:
        st.session_state.conn = load_into_duckdb()
        st.session_state.schema = describe_schema(st.session_state.conn)
        # Build RAG index once per session
        try:
            docs = build_schema_docs(st.session_state.conn)
            st.session_state.rag_index = build_rag_index(docs)
        except Exception:
            st.session_state.rag_index = None
        st.success("Loaded into DuckDB")
    except Exception as e:
        st.error(str(e))

mem = st.session_state.get("mem") or ConversationMemory()
st.session_state.mem = mem

# Model Selection
st.sidebar.markdown("---")
st.sidebar.subheader("AI Model Selection")

try:
    available_providers = get_available_providers()
    provider_names = {
        "gemini": "Google Gemini 2.5 Flash",
        "deepseek": "DeepSeek Chat",
        "openai": "OpenAI GPT-4o Mini",
        "openrouter": "OpenRouter (Claude 3.5 Sonnet)",
        "auto": "Auto (Smart Fallback)"
    }
    
    # Add auto option if multiple providers available
    if len(available_providers) > 1:
        available_providers = ["auto"] + available_providers
    
    selected_provider = st.sidebar.selectbox(
        "Choose AI Provider",
        available_providers,
        format_func=lambda x: provider_names.get(x, x.title()),
        key="ai_provider"
    )
    
    # Store selected provider in session state
    st.session_state.selected_provider = selected_provider
    
    # Display current model info
    if selected_provider == "gemini":
        model_info = os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash")
    elif selected_provider == "deepseek":
        model_info = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    elif selected_provider == "openai":
        model_info = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    elif selected_provider == "openrouter":
        model_info = os.getenv("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet")
    else:
        model_info = "Auto-selection"
    
    st.sidebar.caption(f"Model: {model_info}")
    
except Exception:
    st.sidebar.error("No AI providers available. Check API keys in Token.txt")
    st.session_state.selected_provider = "auto"

# Clear any cached resolved model and use our configured model
if "GEMINI_RESOLVED_MODEL" in os.environ:
    del os.environ["GEMINI_RESOLVED_MODEL"]
timeout_s = os.getenv("GEMINI_TIMEOUT", "45")
st.caption(f"Provider: {st.session_state.get('selected_provider', 'auto')}  •  Timeout: {timeout_s}s")

# Sample questions for RAG search
st.markdown("### Sample Queries")
rag_samples = [
    "Top 5 cities by customer density",
    "Show monthly revenue trend for São Paulo",
    "Compare customer growth in Rio vs. Belo Horizonte",
    "Which regions have the highest market penetration?",
    "Average order value by state",
]
cols = st.columns(len(rag_samples))
for i, q in enumerate(rag_samples):
    if cols[i].button(q, key=f"rag_sample_{i}"):
        st.session_state["prefill_q"] = q

# Sidebar: previous questions
st.sidebar.markdown("---")
st.sidebar.subheader("Previous questions")
history = st.session_state.get("history", [])
prev_qs = [m[0] for m in mem.buf]
if prev_qs:
    for i, q in enumerate(list(prev_qs)[::-1]):
        if st.sidebar.button(q[:60] + ("…" if len(q) > 60 else ""), key=f"reask_{i}"):
            st.session_state["prefill_q"] = q
else:
    st.sidebar.write("No questions yet.")

# Sidebar: controls
st.sidebar.markdown("---")
col_a, col_b = st.sidebar.columns(2)
if col_a.button("Clear chat"):
    st.session_state.chat = []
if col_b.button("Clear memory"):
    st.session_state.mem = ConversationMemory()

# Sidebar: Visual Query Builder (no LLM)
st.sidebar.markdown("---")
with st.sidebar.expander("Visual Query Builder", expanded=False):
    if st.session_state.get("conn"):
        try:
            tables = [r[0] for r in st.session_state.conn.execute("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema='main' ORDER BY table_name
            """).fetchall()]
        except Exception:
            tables = []
        sel_table = st.selectbox("Table", options=tables, key="vq_table")
        cols = []
        num_cols = []
        if sel_table:
            try:
                info = st.session_state.conn.execute(f"PRAGMA table_info({sel_table})").df()
                cols = list(info["name"]) if "name" in info.columns else []
                # crude numeric detection via a peek
                sample = st.session_state.conn.execute(f"SELECT * FROM {sel_table} LIMIT 100").df()
                num_cols = list(sample.select_dtypes(include=["number"]).columns)
            except Exception:
                pass
        group_col = st.selectbox("Group by", options=cols or [""], key="vq_group")
        metric = st.selectbox("Metric", ["Count rows", "Sum", "Average"], index=0, key="vq_metric")
        metric_col = None
        if metric != "Count rows":
            metric_col = st.selectbox("Column", options=num_cols or cols or [""], key="vq_metric_col")
        top_n = st.slider("Top N", 5, 100, 10, key="vq_topn")
        if st.button("Run", key="vq_run") and sel_table and group_col:
            agg = "COUNT(*)" if metric == "Count rows" else ("SUM(" + metric_col + ")" if metric == "Sum" else "AVG(" + metric_col + ")")
            sql = f"SELECT {group_col} AS x, {agg} AS y FROM {sel_table} GROUP BY 1 ORDER BY 2 DESC LIMIT {top_n}"
            try:
                df_vq = st.session_state.conn.execute(sql).df()
                st.session_state.chat.append({"role": "user", "content": f"[VQB] {sel_table} by {group_col} ({metric})"})
                with st.container():
                    st.markdown("Generated via Visual Query Builder")
                    tabs = st.tabs(["Table", "Chart", "SQL"])
                    with tabs[0]:
                        st.dataframe(df_vq, use_container_width=True)
                    with tabs[1]:
                        if df_vq.shape[1] >= 2:
                            try:
                                fig = px.bar(df_vq, x=df_vq.columns[0], y=df_vq.columns[1])
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception:
                                st.info("Unable to render chart.")
                    with tabs[2]:
                        st.code(sql, language="sql")
                st.session_state.chat.append({"role": "assistant", "content": "Visual query executed."})
            except Exception as e:
                st.sidebar.error(str(e))

# Main Dashboard Layout
st.markdown("---")
st.subheader("Geospatial Customer Density Dashboard")

if st.session_state.get("conn"):
    # Sidebar Controls for Map
    st.sidebar.markdown("---")
    st.sidebar.subheader("Map Controls")
    
    # Region selector
    regions = {
        "All Brazil": "",
        "North": "AC,AM,AP,PA,RO,RR,TO",
        "Northeast": "AL,BA,CE,MA,PI,PE,RN,SE",
        "Southeast": "ES,MG,RJ,SP",
        "South": "PR,RS,SC",
        "Central-West": "DF,GO,MT,MS"
    }
    selected_region = st.sidebar.selectbox("Region", list(regions.keys()), key="map_region")
    
    # Density threshold
    min_customers = st.sidebar.slider("Min Customers", 10, 1000, 50, key="min_customers")
    
    # Zoom level for dynamic labeling
    zoom_level = st.sidebar.slider("Label Detail (Zoom)", 1, 10, 5, key="zoom_level",
                                  help="Higher values show more city labels")
    
    # Main layout: Map (70%) + Analytics Panel (30%)
    map_col, analytics_col = st.columns([7, 3])
    
    with map_col:
        try:
            # Build query with region filter
            if regions[selected_region]:
                state_list = [s.strip() for s in regions[selected_region].split(',')]
                state_quotes = ["'" + state + "'" for state in state_list]
                region_filter = "AND c.customer_state IN (" + ",".join(state_quotes) + ")"
            else:
                region_filter = ""
            
            df_geo = st.session_state.conn.execute(
                f"""
                SELECT c.customer_city AS city,
                       c.customer_state AS state,
                       AVG(g.geolocation_lat) AS lat,
                       AVG(g.geolocation_lng) AS lon,
                       COUNT(DISTINCT c.customer_unique_id) AS customers,
                       SUM(oi.price) AS revenue
                FROM olist_customers_dataset c
                JOIN olist_geolocation_dataset g
                  ON g.geolocation_zip_code_prefix = c.customer_zip_code_prefix
                JOIN olist_orders_dataset o ON c.customer_id = o.customer_id
                JOIN olist_order_items_dataset oi ON o.order_id = oi.order_id
                WHERE 1=1 {region_filter}
                GROUP BY 1,2
                HAVING customers >= {min_customers}
                ORDER BY customers DESC
                """
            ).df()
            
            if not df_geo.empty:
                # Dynamic city labeling based on zoom
                def get_visible_cities(df, zoom):
                    if zoom <= 3:
                        # Show only top 10 major cities
                        return df.head(10)
                    elif zoom <= 6:
                        # Show top 50 cities
                        return df.head(50)
                    elif zoom <= 8:
                        # Show top 100 cities
                        return df.head(100)
                    else:
                        # Show all filtered cities
                        return df
                
                df_plot = get_visible_cities(df_geo, zoom_level)
                
                # Create interactive map with custom styling and click events
                figg = px.scatter_geo(
                    df_plot,
                    lat="lat",
                    lon="lon",
                    size="customers",
                    hover_data={
                        "city": True,
                        "state": True,
                        "customers": ":,",
                        "revenue": "$,.0f"
                    },
                    projection="natural earth",
                    scope="south america",
                    title=f"Customer Density - {selected_region}",
                    custom_data=["city", "state", "customers", "revenue"]
                )
                
                # Dynamic text labels
                if zoom_level <= 3:
                    # Only show major city names
                    figg.update_traces(
                        text=df_plot.head(10)["city"].apply(lambda x: str(x).title()),
                        textposition="top center",
                        textfont=dict(color="#e5e7eb", size=11),
                        marker=dict(color="#38bdf8", opacity=0.6, line=dict(width=0)),
                        hovertemplate="<b>%{customdata[0]}</b><br>" +
                                     "State: %{customdata[1]}<br>" +
                                     "Customers: %{customdata[2]:,}<br>" +
                                     "Revenue: $%{customdata[3]:,.0f}<br>" +
                                     "<extra></extra>"
                    )
                elif zoom_level <= 6:
                    # Show city + customer count
                    figg.update_traces(
                        text=df_plot.head(50).apply(
                            lambda r: f"{str(r['city']).title()}\n{int(r['customers'])}", axis=1
                        ),
                        textposition="top center",
                        textfont=dict(color="#e5e7eb", size=9),
                        marker=dict(color="#38bdf8", opacity=0.55, line=dict(width=0)),
                        hovertemplate="<b>%{customdata[0]}</b><br>" +
                                     "State: %{customdata[1]}<br>" +
                                     "Customers: %{customdata[2]:,}<br>" +
                                     "Revenue: $%{customdata[3]:,.0f}<br>" +
                                     "<extra></extra>"
                    )
                else:
                    # Show city + customers + revenue
                    figg.update_traces(
                        text=df_plot.apply(
                            lambda r: f"{str(r['city']).title()}\n{int(r['customers'])}\n${r['revenue']:,.0f}", axis=1
                        ),
                        textposition="top center",
                        textfont=dict(color="#e5e7eb", size=8),
                        marker=dict(color="#38bdf8", opacity=0.5, line=dict(width=0)),
                        hovertemplate="<b>%{customdata[0]}</b><br>" +
                                     "State: %{customdata[1]}<br>" +
                                     "Customers: %{customdata[2]:,}<br>" +
                                     "Revenue: $%{customdata[3]:,.0f}<br>" +
                                     "<extra></extra>"
                    )
                
                figg.update_geos(
                    fitbounds="locations",
                    landcolor="#0f172a",
                    bgcolor="#0b1220",
                    showcountries=True,
                    showcoastlines=True,
                    showframe=True,
                    coastlinecolor="#38bdf8",
                    countrycolor="#38bdf8",
                    framecolor="#38bdf8"
                )
                figg.update_layout(
                    height=700,
                    margin=dict(l=0, r=0, t=50, b=0),
                    showlegend=False,
                    paper_bgcolor="#0b1220",
                    plot_bgcolor="#0b1220"
                )
                
                # Display map without click events for now
                st.plotly_chart(figg, use_container_width=True, key="geo_map")
                
            else:
                st.info("No geospatial data available for current filters.")
                
        except Exception as e:
            st.info(f"Geospatial view unavailable: {str(e)}")
    
    with analytics_col:
        st.markdown("### Analytics Panel")
        
        # Top cities by customer density
        st.markdown("**Top Cities by Customers**")
        try:
            df_top_cities = st.session_state.conn.execute(
                f"""
                SELECT c.customer_city AS city,
                       COUNT(DISTINCT c.customer_unique_id) AS customers
                FROM olist_customers_dataset c
                {'JOIN olist_orders_dataset o ON c.customer_id = o.customer_id '
                 'JOIN olist_geolocation_dataset g ON g.geolocation_zip_code_prefix = c.customer_zip_code_prefix '
                 f'WHERE g.geolocation_lat BETWEEN -34 AND 6 AND g.geolocation_lng BETWEEN -74 AND -32 '
                 f'{region_filter.replace("c.customer_state", "c.customer_state")} '
                 if regions[selected_region] else ''}
                GROUP BY 1
                ORDER BY customers DESC
                LIMIT 10
                """
            ).df()
            
            if not df_top_cities.empty:
                fig_bar = px.bar(
                    df_top_cities.head(7),
                    x="customers",
                    y="city",
                    orientation="h",
                    title="Top 7 Cities"
                )
                fig_bar.update_layout(
                    height=300,
                    margin=dict(l=10, r=10, t=40, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e5e7eb")
                )
                fig_bar.update_xaxes(showgrid=False)
                fig_bar.update_yaxes(showgrid=False)
                st.plotly_chart(fig_bar, use_container_width=True)
        except Exception:
            st.info("City analytics unavailable")
        
        # Regional comparison
        st.markdown("**Regional Overview**")
        try:
            df_regional = st.session_state.conn.execute(
                """
                SELECT 
                    CASE 
                        WHEN c.customer_state IN ('AC','AM','AP','PA','RO','RR','TO') THEN 'North'
                        WHEN c.customer_state IN ('AL','BA','CE','MA','PI','PE','RN','SE') THEN 'Northeast'
                        WHEN c.customer_state IN ('ES','MG','RJ','SP') THEN 'Southeast'
                        WHEN c.customer_state IN ('PR','RS','SC') THEN 'South'
                        WHEN c.customer_state IN ('DF','GO','MT','MS') THEN 'Central-West'
                    END AS region,
                    COUNT(DISTINCT c.customer_unique_id) AS customers,
                    SUM(oi.price) AS revenue
                FROM olist_customers_dataset c
                JOIN olist_orders_dataset o ON c.customer_id = o.customer_id
                JOIN olist_order_items_dataset oi ON o.order_id = oi.order_id
                GROUP BY 1
                ORDER BY customers DESC
                """
            ).df()
            
            if not df_regional.empty:
                for _, row in df_regional.iterrows():
                    st.markdown(f"""
                    <div style="background: rgba(255,255,255,0.05); padding: 0.5rem; border-radius: 5px; margin-bottom: 0.5rem;">
                        <strong>{row['region']}</strong><br>
                        <small>Customers: {row['customers']:,} | Revenue: ${row['revenue']:,.0f}</small>
                    </div>
                    """, unsafe_allow_html=True)
        except Exception:
            st.info("Regional data unavailable")
        
        # Market penetration indicator
        st.markdown("**Market Penetration**")
        try:
            total_population = 214000000  # Approximate Brazil population
            active_customers = st.session_state.conn.execute(
                "SELECT COUNT(DISTINCT customer_unique_id) FROM olist_customers_dataset"
            ).fetchone()[0]
            penetration = (active_customers / total_population) * 100
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 1rem; border-radius: 8px; text-align: center; color: white;">
                <div style="font-size: 1.5rem; font-weight: bold;">{penetration:.3f}%</div>
                <div style="font-size: 0.9rem; opacity: 0.9;">Market Penetration</div>
            </div>
            """, unsafe_allow_html=True)
        except Exception:
            st.info("Penetration metrics unavailable")
else:
    st.info("Load dataset to view geospatial dashboard")

# Chat UI
st.markdown("---")
chat_container = st.container()
if "chat" not in st.session_state:
    st.session_state.chat = []
if "llm_cache" not in st.session_state:
    st.session_state.llm_cache = {}

for message in st.session_state.chat:
    with chat_container.chat_message(message["role"]):
        st.markdown(message["content"])

prefill = st.session_state.pop("prefill_q", None)
user_input = st.chat_input(placeholder="Ask a question about the Olist data", key="chat_input")
if prefill and not user_input:
    user_input = prefill

if user_input:
    if not st.session_state.get("conn"):
        st.error("Dataset not loaded. Download and Load into DuckDB first.")
    else:
        st.session_state.chat.append({"role": "user", "content": user_input})
        with chat_container.chat_message("user"):
            st.markdown(user_input)
        with chat_container.chat_message("assistant"):
            # Cache key based on question + schema hash
            cache_key = hashlib.sha256((user_input + "\n" + st.session_state.schema).encode("utf-8")).hexdigest()[:16]
            cached = st.session_state.llm_cache.get(cache_key)
            # Simple per-session rate limiter for LLM
            now = time.time()
            min_interval = 25
            last = st.session_state.get("last_llm", 0)
            if cached is None and (now - last) < min_interval:
                wait = int(min_interval - (now - last))
                st.warning(f"Rate limit: please wait ~{wait}s before the next AI query, or use the Visual Query Builder / Quick Insights.")
            with st.spinner("Thinking..."):
                try:
                    # Compose hint with optional RAG context
                    hint = mem.hint()
                    if st.session_state.get("rag_hint"):
                        hint = (hint + "\n\n" + st.session_state.rag_hint).strip()
                        # one-shot usage then clear
                        st.session_state.rag_hint = None
                    if cached is None:
                        selected_provider = st.session_state.get("selected_provider", "auto")
                        df, sql, analysis = answer_question(user_input, st.session_state.conn, st.session_state.schema, hint, selected_provider)
                        st.session_state.last_llm = time.time()
                        st.session_state.llm_cache[cache_key] = {"df": df, "sql": sql, "analysis": analysis}
                        # keep cache small
                        if len(st.session_state.llm_cache) > 30:
                            st.session_state.llm_cache.pop(next(iter(st.session_state.llm_cache)))
                    else:
                        df, sql, analysis = cached["df"], cached["sql"], cached["analysis"]
                    mem.note(user_input, analysis)
                    # KPI band
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("Rows", f"{len(df):,}")
                    k2.metric("Columns", f"{df.shape[1]:,}")
                    num_cols = list(df.select_dtypes(include=["number"]).columns)
                    k3.metric("Numeric cols", f"{len(num_cols)}")
                    k4.metric("Missing %", f"{(df.isna().sum().sum() / (df.shape[0]*max(df.shape[1],1) or 1)) * 100:.1f}%")

                    tabs = st.tabs(["Summary", "Table", "Chart", "Export", "SQL"])
                    with tabs[0]:
                        st.markdown(analysis)
                        # smart suggestions (no extra LLM calls)
                        sugg = []
                        low = user_input.lower()
                        if "elect" in low:
                            sugg = [
                                "Top performing electronics sub-categories",
                                "Electronics return rates vs other categories",
                                "Seasonal trends in electronics purchases",
                            ]
                        elif "category" in low:
                            sugg = [
                                "Categories with declining sales but high ratings",
                                "Category-wise repeat purchase rate",
                                "Quarterly category growth trend",
                            ]
                        if sugg:
                            st.markdown("**You might also ask:**")
                            scols = st.columns(len(sugg))
                            for i, s in enumerate(sugg):
                                if scols[i].button(s, key=f"sugg_{i}"):
                                    st.session_state["prefill_q"] = s
                    with tabs[1]:
                        st.dataframe(df, use_container_width=True)
                    last_fig = None
                    with tabs[2]:
                        if df.shape[1] >= 2:
                            x = st.selectbox("X-axis", options=list(df.columns), index=0, key="chart_x")
                            y = st.selectbox("Y-axis", options=[c for c in df.columns if c != x], index=0 if df.shape[1] > 1 else None, key="chart_y")
                            ctype = st.selectbox("Chart type", ["bar", "line", "scatter", "pie"], index=0, key="chart_type")
                            fig = None
                            try:
                                if ctype == "bar":
                                    fig = px.bar(df.head(200), x=x, y=y)
                                elif ctype == "line":
                                    fig = px.line(df.head(200), x=x, y=y)
                                elif ctype == "scatter":
                                    fig = px.scatter(df.head(200), x=x, y=y)
                                elif ctype == "pie":
                                    fig = px.pie(df.head(200), names=x, values=y)
                                if fig is not None:
                                    st.plotly_chart(fig, use_container_width=True)
                                    last_fig = fig
                            except Exception:
                                st.info("Unable to render chart for the current selection.")
                        else:
                            st.info("Need at least two columns to chart.")
                    with tabs[3]:
                        st.download_button("Download results as CSV", data=df.to_csv(index=False).encode("utf-8"), file_name="query_results.csv", mime="text/csv")
                        # PDF export (table head + chart image if available)
                        try:
                            from fpdf import FPDF
                            if st.button("Export PDF", key="export_pdf"):
                                pdf = FPDF(orientation="P", unit="mm", format="A4")
                                pdf.add_page()
                                pdf.set_font("Arial", "B", 16)
                                pdf.cell(0, 10, "InsightDock Report", ln=1)
                                pdf.set_font("Arial", size=11)
                                pdf.multi_cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\nQuestion: {user_input}")
                                if last_fig is not None:
                                    img_bytes = last_fig.to_image(format="png")
                                    img_stream = io.BytesIO(img_bytes)
                                    img_path = "_temp_chart.png"
                                    with open(img_path, "wb") as f:
                                        f.write(img_stream.getvalue())
                                    pdf.image(img_path, w=180)
                                # Table head
                                head = df.head(25)
                                pdf.ln(4)
                                pdf.set_font("Arial", "B", 12)
                                pdf.cell(0, 8, "Results (first 25 rows)", ln=1)
                                pdf.set_font("Arial", size=9)
                                for _, row in head.iterrows():
                                    pdf.multi_cell(0, 6, ", ".join(map(str, row.values)))
                                pdf_bytes = bytes(pdf.output(dest='S').encode('latin1'))
                                st.download_button("Download PDF", data=pdf_bytes, file_name="insightdock_report.pdf", mime="application/pdf")
                        except Exception:
                            pass
                    with tabs[4]:
                        st.code(sql, language="sql")
                    st.session_state.chat.append({"role": "assistant", "content": analysis})
                except Exception as e:
                    st.error(str(e))

st.divider()
with st.expander("Schema"):
    st.code(st.session_state.schema or "", language="text")
