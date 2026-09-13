# Hugging Face Docker Space: https://huggingface.co/docs/hub/spaces-sdks-docker
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/huggingface

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . ./

# Download both retrieval models during image build. The application defaults
# to local-only loading at runtime, eliminating cold-start Hub traffic.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('BAAI/bge-small-en-v1.5'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Spaces runs containers as a non-root user. Model cache and source must remain
# readable after the user switch.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app /opt/huggingface
USER appuser

EXPOSE 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
