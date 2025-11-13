# Deployment Guide

## Quick Start

1. **Setup configuration**
```bash
python setup.py --install-deps
```

2. **Configure API keys**
Edit `Token.txt` with your actual API keys (at least one required):
- GEMINI_API_KEY (recommended)
- OPENAI_API_KEY
- OPENROUTER_API_KEY
- DEEPSEEK_API_KEY

3. **Configure Kaggle**
Edit `kaggle.json` with your Kaggle credentials

4. **Run the application**
```bash
streamlit run streamlit_app.py
```

## Environment Variables

Alternatively, you can set environment variables instead of using Token.txt:
```bash
export GEMINI_API_KEY="your_key_here"
export OPENAI_API_KEY="your_key_here"
# etc.
```

## Docker Deployment (Optional)

Create a Dockerfile:
```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t insightdock .
docker run -p 8501:8501 -e GEMINI_API_KEY="your_key" insightdock
```

## Cloud Deployment

### Streamlit Cloud
1. Connect your GitHub repository
2. Add secrets in Streamlit Cloud dashboard
3. Deploy automatically

### Heroku
1. Add `Procfile`: `web: streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0`
2. Set environment variables in Heroku dashboard
3. Deploy via Git

## Security Notes

- Never commit API keys to version control
- Use environment variables in production
- Token.txt and kaggle.json are gitignored for security
