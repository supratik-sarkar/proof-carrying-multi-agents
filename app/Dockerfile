# Optional Docker deployment for HF Spaces.
# Use this if you want full control over the runtime — set sdk: docker
# in README.md frontmatter instead of sdk: streamlit.
#
# For most cases the streamlit SDK is fine and faster to deploy.

FROM python:3.11-slim

WORKDIR /app

# System deps (minimal — keeping image small for fast cold-starts)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python deps first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App
COPY . .

# HF Spaces expects the app on port 7860
ENV PORT=7860
EXPOSE 7860

CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
