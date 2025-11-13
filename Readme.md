# 🇧🇷 InsightDock - Brazil Customer Analytics

> Transform Brazilian e-commerce data into actionable insights with AI-powered natural language queries

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 What is InsightDock?

InsightDock is an **intelligent analytics platform** that lets you explore Brazilian e-commerce data using **plain English questions**. No SQL knowledge required!

### ✨ Key Highlights
- 🤖 **Ask in English** → Get instant SQL + visualizations
- 🗺️ **Interactive Brazil Map** → Explore customer density by city
- ⚡ **Lightning Fast** → DuckDB processes millions of records in seconds
- 🔄 **4 AI Providers** → Never worry about API downtime
- 📊 **Real-time Dashboard** → Live metrics and regional insights




https://github.com/user-attachments/assets/63f25f10-e8f2-43eb-a97d-dbe2426b511e

---

## 🚀 Quick Start (3 Steps!)

### Step 1: Get the Code
```bash
git clone https://github.com/Nachiket1234/InsightDock.git
cd InsightDock
```

### Step 2: Setup & Install
```bash
# Install dependencies
pip install -r requirements.txt

# Setup configuration files
python setup.py --install-deps
```

### Step 3: Add Your API Keys
Edit `Token.txt` with at least one AI provider key:
```
GEMINI_API_KEY=your_key_here        # 👈 Recommended (Free tier available)
OPENAI_API_KEY=your_key_here        # 👈 Alternative
OPENROUTER_API_KEY=your_key_here    # 👈 Premium models
```

### Step 4: Launch! 🎉
```bash
streamlit run streamlit_app.py
```
Open http://localhost:8501 and start exploring!

---

## 💡 How It Works

### 1. 🗣️ Ask Questions in Plain English
```
"Top 5 cities by customer density"
"Show monthly revenue trend for São Paulo"
"Which regions have the highest market penetration?"
```

### 2. 🤖 AI Converts to SQL
The system automatically generates optimized SQL queries from your questions.

### 3. 📊 Get Interactive Results
- **Tables** with sortable data
- **Charts** with Plotly visualizations  
- **Maps** showing geographic insights
- **Analysis** with business recommendations

---

## 🎮 Demo Walkthrough

### Load Brazilian E-commerce Data
1. Click **"Load into DuckDB"** in sidebar
2. System downloads 100,000+ Olist records automatically
3. Data processed in 50,000 record chunks for optimal performance

### Explore with AI
1. Choose your AI provider (Gemini recommended)
2. Type questions like: *"Compare customer growth in Rio vs. Belo Horizonte"*
3. Get instant SQL, charts, and business insights

### Interactive Brazil Map
- 🔍 **Zoom levels 1-10** → More cities appear as you zoom in
- 🎯 **Click regions** → Filter by North, Northeast, Southeast, South, Central-West
- 💰 **Bubble sizes** → Represent customer density
- 📍 **City labels** → Show names and customer counts

---

## 🛠️ Technology Stack

<table>
<tr>
<td><strong>🎨 Frontend</strong></td>
<td>Streamlit + Custom CSS + Inter Font</td>
</tr>
<tr>
<td><strong>🧠 AI/ML</strong></td>
<td>Google Gemini, OpenAI, OpenRouter, DeepSeek</td>
</tr>
<tr>
<td><strong>💾 Database</strong></td>
<td>DuckDB (In-memory analytics)</td>
</tr>
<tr>
<td><strong>📊 Visualization</strong></td>
<td>Plotly Express (Interactive charts & maps)</td>
</tr>
<tr>
<td><strong>🔍 Search</strong></td>
<td>LangChain + FAISS (Vector search)</td>
</tr>
</table>

---

## 🎯 Key Features Explained

### 🤖 Multi-AI Provider System
Never get stuck with API downtime! The system tries providers in this order:
1. **Gemini** (Fast & reliable)
2. **OpenAI** (Cost-effective)  
3. **OpenRouter** (Premium models)
4. **DeepSeek** (Budget option)

### 🗺️ Smart Geospatial Analytics
- **Dynamic Labels**: Show 10 cities at zoom 1-3, up to 100+ cities at zoom 10
- **Regional Filters**: Focus on specific Brazilian regions
- **Density Visualization**: Bubble size = customer concentration

### ⚡ Performance Optimizations
- **Memory Management**: 50K record chunks prevent crashes
- **Smart Caching**: Repeated queries load instantly
- **Rate Limiting**: Intelligent API quota management

---

## 📋 Prerequisites

| Requirement | Details |
|-------------|---------|
| **Python** | 3.12+ (3.8+ works but 3.12 recommended) |
| **API Keys** | At least 1 AI provider (see options below) |
| **Kaggle** | Account for dataset access (free) |
| **Memory** | 4GB+ RAM recommended for large datasets |

### 🔑 AI Provider Options

| Provider | Cost | Speed | Quality | Free Tier |
|----------|------|-------|---------|-----------|
| **Google Gemini** | 💰 | ⚡⚡⚡ | ⭐⭐⭐⭐ | ✅ Yes |
| **OpenAI** | 💰💰 | ⚡⚡ | ⭐⭐⭐⭐⭐ | ❌ No |
| **OpenRouter** | 💰💰💰 | ⚡⚡ | ⭐⭐⭐⭐⭐ | ❌ No |
| **DeepSeek** | 💰 | ⚡ | ⭐⭐⭐ | ✅ Yes |

---

## 🗂️ Project Structure

```
InsightDock/
├── 🎯 streamlit_app.py          # Main application (your starting point)
├── 🤖 hybrid_llm.py            # Multi-AI provider system  
├── 📊 data_loader.py           # Dataset loading & processing
├── 🔄 sql_agent.py             # Natural language → SQL magic
├── 🧠 memory.py                # Conversation context
├── 🔍 rag.py                   # Vector search system
├── 🔐 app_secrets.py           # Secure config loader
├── 📦 requirements.txt         # Python dependencies
├── 📖 README.md               # This file!
├── ⚙️ setup.py                # Easy setup script
├── 🛡️ .gitignore              # Security protection
├── 📝 Token.txt.template      # API key template
└── 🗃️ data/                   # Dataset storage (auto-created)
```

---

## 🎨 Sample Queries to Try

### 📈 Business Intelligence
```
"Top 10 cities by revenue"
"Monthly sales trend for 2018"
"Average order value by state"
"Customer retention rate by region"
```

### 🗺️ Geographic Analysis  
```
"Show customer density across Brazil"
"Which states have the most orders?"
"Compare São Paulo vs Rio de Janeiro performance"
"Market penetration in Northeast region"
```

### 🛍️ Product Insights
```
"Most popular product categories"
"Seasonal trends in electronics sales"
"Products with highest ratings"
"Category performance by region"
```

---

## 🔧 Advanced Configuration

### Environment Variables (Alternative to Token.txt)
```bash
export GEMINI_API_KEY="your_key_here"
export OPENAI_API_KEY="your_key_here"
export OPENROUTER_API_KEY="your_key_here"
```

### Custom Model Selection
```
# In Token.txt, specify exact models:
GEMINI_MODEL=models/gemini-2.5-flash
OPENAI_MODEL=gpt-4o-mini
OPENROUTER_MODEL=anthropic/claude-3.5-sonnet
```

### Performance Tuning
```
# Adjust timeouts and chunk sizes:
GEMINI_TIMEOUT=60
CHUNK_SIZE=25000  # Reduce if you have memory issues
```

---

## 🚀 Deployment Options

### 🏠 Local Development
```bash
streamlit run streamlit_app.py
```

### 🐳 Docker
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501"]
```

### ☁️ Cloud Platforms
- **Streamlit Cloud**: Connect GitHub → Deploy automatically
- **Heroku**: Add `Procfile` → Set environment variables
- **Railway**: One-click deployment from GitHub

---

## 🤝 Contributing

We love contributions! Here's how to help:

1. **🍴 Fork** the repository
2. **🌿 Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **💾 Commit** your changes (`git commit -m 'Add AmazingFeature'`)
4. **📤 Push** to the branch (`git push origin feature/AmazingFeature`)
5. **🔄 Open** a Pull Request

### 🐛 Found a Bug?
Open an issue with:
- Steps to reproduce
- Expected vs actual behavior
- Your Python version and OS

---

## 📞 Support & Community

- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/Nachiket1234/InsightDock/issues)
- 💡 **Feature Requests**: [GitHub Discussions](https://github.com/Nachiket1234/InsightDock/discussions)
- 📧 **Direct Contact**: Open an issue and we'll respond quickly!

---

## 🙏 Acknowledgments

- 🇧🇷 **Olist** for the amazing Brazilian e-commerce dataset
- 🚀 **Streamlit** for the incredible framework
- 🤖 **Google, OpenAI, OpenRouter, DeepSeek** for AI capabilities
- 🦆 **DuckDB** for lightning-fast analytics
- 💙 **Open Source Community** for inspiration and support

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**🎉 Built with ❤️ for the Brazilian e-commerce analytics community**

[⭐ Star this repo](https://github.com/Nachiket1234/InsightDock) • [🐛 Report Bug](https://github.com/Nachiket1234/InsightDock/issues) • [💡 Request Feature](https://github.com/Nachiket1234/InsightDock/discussions)

</div>
