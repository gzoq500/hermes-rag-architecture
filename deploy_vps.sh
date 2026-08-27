#!/bin/bash
set -e
printf "=== DEPLOYING REPO KE VPS 4 ===\n"

# Setup env file
cat << "ENVEOF" > /etc/hermes-rag-gateway.env
RAG_GATEWAY_HOST=0.0.0.0
RAG_GATEWAY_PORT=20128
ROUTER_BASE_URL=http://127.0.0.1:20130
GEMINI_EMBEDDING_MODEL=models/gemini-embedding-2
GEMINI_EMBEDDING_DIMENSIONS=3072
ZILLIZ_COLLECTION=hermes_gemini_memory
RAG_RECENT_MESSAGES=24
RAG_RETRIEVAL_LIMIT=3
RAG_RETRIEVAL_MIN_CHARS=10
RAG_CHUNK_MAX_CHARS=8000
RAG_MINIMUM_INGEST_CHARS=15
ENVEOF

# Copy over extracted secrets
cat /root/.rag_secrets_backup.txt | grep ZILLIZ | sed 's/MILVUS_TOKEN *= */ZILLIZ_TOKEN=/' | tr -d '"' | tr -d "'" >> /etc/hermes-rag-gateway.env
cat /root/.rag_secrets_backup.txt | grep GEMINI | sed 's/GEMINI_API_KEY *= */GEMINI_API_KEY=/' | tr -d '"' | tr -d "'" >> /etc/hermes-rag-gateway.env
echo "ZILLIZ_URI=https://in01-8fb19767351ff95.aws-ap-southeast-1.vectordb.zillizcloud.com:19532" >> /etc/hermes-rag-gateway.env
chmod 600 /etc/hermes-rag-gateway.env

# Checkout target deployment path
mkdir -p /opt/hermes-rag-architecture
cd /opt/hermes-rag-architecture

# Clone/pull from Github
if [ -d .git ]; then
    git fetch origin main
    git reset --hard origin/main
else
    git clone https://github.com/gzoq500/hermes-rag-architecture.git .
fi

# Build Virtualenv
apt-get update && apt-get install -y python3-venv python3-pip
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Apply Systemd Services
cp src/rag-worker.service /etc/systemd/system/rag-worker.service
cp src/9router-shifted.service /etc/systemd/system/9router.service

systemctl daemon-reload
systemctl enable rag-worker.service

systemctl restart 9router.service
systemctl restart rag-worker.service

# Setup the plugin for Hermes Agent
mkdir -p /root/.hermes/plugins
cp -r plugins/hermes-rag-session-header /root/.hermes/plugins/
# Tell hermes to enable the plugin
if command -v hermes &> /dev/null; then
    hermes plugins enable hermes-rag-session-header --no-allow-tool-override || echo "Failed to enable plugin (maybe already enabled or missing Hermes binary)"
fi

echo "Deployment complete."
