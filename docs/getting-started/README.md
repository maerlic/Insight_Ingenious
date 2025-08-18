---
title: "Quick Start Guide"
layout: single
permalink: /getting-started/
sidebar:
  nav: "docs"
toc: true
toc_label: "Quick Start Steps"
toc_icon: "rocket"
---

## Quick Start

Get up and running in 5 minutes with Azure OpenAI!

### Prerequisites
- Python 3.13 or higher (required — earlier versions are not supported)
- Azure OpenAI resource with at least one **chat** model deployment and one **embedding** model deployment
- (Optional) Azure AI Search, Azure SQL, and Azure Storage if you want the full cloud setup
- [uv package manager](https://docs.astral.sh/uv/)

---

## 5-Minute Setup

### 1) Install and Initialize
```bash
# Navigate to your desired project directory first
cd /path/to/your/project

# Set up the uv project
uv init

# Choose installation based on features needed
uv add "ingenious[azure-full]"   # Recommended: Full Azure integration (core, auth, azure, ai, database, ui)
# OR
uv add "ingenious[standard]"     # For local testing: includes SQL agent support (core, auth, ai, database)

# Initialize project template in the current directory (adds sample workflows like bike-insights)
uv run ingen init
```

---

### 2) Configure Credentials (.env)

Create a `.env` file **in the project root** and paste the **dummy template** below. Replace the placeholders with your real values.

> **Important:** These variables reflect the current configuration loader.
> - Azure Search is configured via **`azure_search_services`** (top‑level).
> - Booleans should be the strings **`true`** or **`false`** (avoid `1/0`).

#### 📄 Minimal Local-Only (no Azure AI Search)
If you only want to test chat with local history:
```bash
# --- Web server ---
INGENIOUS_WEB_CONFIGURATION__IP_ADDRESS=0.0.0.0
INGENIOUS_WEB_CONFIGURATION__PORT=8000

# --- Chat service ---
INGENIOUS_CHAT_SERVICE__TYPE=multi_agent

# --- Models (Azure OpenAI - chat only) ---
INGENIOUS_MODELS__0__MODEL=gpt-4.1-mini
INGENIOUS_MODELS__0__API_TYPE=rest
INGENIOUS_MODELS__0__API_VERSION=2024-12-01-preview
INGENIOUS_MODELS__0__DEPLOYMENT=my-gpt-41-mini-deployment
INGENIOUS_MODELS__0__API_KEY=aoai-key-REPLACE_ME
INGENIOUS_MODELS__0__BASE_URL=https://my-aoai-resource.openai.azure.com/
INGENIOUS_MODELS__0__AUTHENTICATION_METHOD=token

# --- Chat history (local SQLite) ---
INGENIOUS_CHAT_HISTORY__DATABASE_TYPE=sqlite
INGENIOUS_CHAT_HISTORY__DATABASE_PATH=./.tmp/chat_history.db
INGENIOUS_CHAT_HISTORY__MEMORY_PATH=./.tmp

# --- Knowledge base (local fallback only) ---
KB_POLICY=local_only
KB_USE_AZURE_SEARCH=false
KB_USE_SEMANTIC_RANKING=false
```

#### ☁️ Full Cloud (Azure OpenAI + Azure AI Search + optional Azure SQL/Storage)
Use this when you want the **knowledge-base-agent** backed by **Azure AI Search**.

```bash
# =========================
# Web server
# =========================
INGENIOUS_WEB_CONFIGURATION__IP_ADDRESS=0.0.0.0
INGENIOUS_WEB_CONFIGURATION__PORT=8000

# =========================
# Chat service
# =========================
INGENIOUS_CHAT_SERVICE__TYPE=multi_agent

# =========================
# Azure OpenAI (chat & embeddings)
# =========================
# Chat (slot 0)
INGENIOUS_MODELS__0__MODEL=gpt-4.1-mini
INGENIOUS_MODELS__0__API_TYPE=rest
INGENIOUS_MODELS__0__API_VERSION=2024-12-01-preview
INGENIOUS_MODELS__0__DEPLOYMENT=my-gpt-41-mini-deployment
INGENIOUS_MODELS__0__API_KEY=aoai-key-REPLACE_ME
INGENIOUS_MODELS__0__BASE_URL=https://my-aoai-resource.openai.azure.com/
INGENIOUS_MODELS__0__AUTHENTICATION_METHOD=token

# Embeddings (slot 1)
# text-embedding-3-small → 1536 dims (text-embedding-3-large → 3072)
INGENIOUS_MODELS__1__MODEL=text-embedding-3-small
INGENIOUS_MODELS__1__API_TYPE=rest
INGENIOUS_MODELS__1__API_VERSION=2024-12-01-preview
INGENIOUS_MODELS__1__DEPLOYMENT=my-embedding-deployment
INGENIOUS_MODELS__1__API_KEY=aoai-key-REPLACE_ME
INGENIOUS_MODELS__1__BASE_URL=https://my-aoai-resource.openai.azure.com/
INGENIOUS_MODELS__1__AUTHENTICATION_METHOD=token

# =========================
# Knowledge Base (Azure AI Search)
# =========================
# Top-level Azure Search service (index 0). Add more services with __1__, __2__, ...
INGENIOUS_AZURE_SEARCH_SERVICES__0__ENDPOINT=https://my-search-service.search.windows.net
INGENIOUS_AZURE_SEARCH_SERVICES__0__KEY=azure-search-key-REPLACE_ME
INGENIOUS_AZURE_SEARCH_SERVICES__0__INDEX_NAME=my-kb-index

# Optional but recommended for best results with semantic ranking:
INGENIOUS_AZURE_SEARCH_SERVICES__0__USE_SEMANTIC_RANKING=true
INGENIOUS_AZURE_SEARCH_SERVICES__0__SEMANTIC_CONFIGURATION_NAME=my-semantic-config

# Index schema mapping (adjust to your index)
INGENIOUS_AZURE_SEARCH_SERVICES__0__ID_FIELD=id
INGENIOUS_AZURE_SEARCH_SERVICES__0__CONTENT_FIELD=content
INGENIOUS_AZURE_SEARCH_SERVICES__0__VECTOR_FIELD=vector
INGENIOUS_AZURE_SEARCH_SERVICES__0__TOP_K_RETRIEVAL=5
INGENIOUS_AZURE_SEARCH_SERVICES__0__TOP_N_FINAL=10

# Knowledge base toggles & policy
KB_USE_AZURE_SEARCH=true          # REQUIRED for Azure-backed KB
KB_USE_SEMANTIC_RANKING=true      # set false if your index lacks semantic config
KB_POLICY=azure_only              # alternatives: prefer_azure | prefer_local | local_only
KB_MODE=direct                    # default mode for KB agent
KB_TOP_K=5
KB_WRITE_CONFIG_SNAPSHOT=true     # optional: writes a snapshot of resolved config for debugging

# =========================
# Chat history (choose ONE)
# =========================
# A) Local SQLite (simple)
INGENIOUS_CHAT_HISTORY__DATABASE_TYPE=sqlite
INGENIOUS_CHAT_HISTORY__DATABASE_PATH=./.tmp/chat_history.db
INGENIOUS_CHAT_HISTORY__MEMORY_PATH=./.tmp

# B) Azure SQL (production)
#INGENIOUS_CHAT_HISTORY__DATABASE_TYPE=azuresql
#INGENIOUS_CHAT_HISTORY__DATABASE_NAME=my-chat-db
#INGENIOUS_CHAT_HISTORY__DATABASE_CONNECTION_STRING=Driver={ODBC Driver 18 for SQL Server};Server=tcp:my-server.database.windows.net,1433;Database=my-chat-db;Uid=my-user;Pwd=my-strong-password;Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;

# =========================
# File storage (optional; for revisions/data over Azure Blob)
# =========================
# Global connection string to your storage account
# Note: you can also use SAS tokens instead of the full connection string.
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=myaccount;AccountKey=REPLACE_ME;EndpointSuffix=core.windows.net

# Revisions (e.g., prompt revisions)
INGENIOUS_FILE_STORAGE__REVISIONS__ENABLE=true
INGENIOUS_FILE_STORAGE__REVISIONS__STORAGE_TYPE=azure
INGENIOUS_FILE_STORAGE__REVISIONS__CONTAINER_NAME=prompts
INGENIOUS_FILE_STORAGE__REVISIONS__PATH=ingenious-files
INGENIOUS_FILE_STORAGE__REVISIONS__ADD_SUB_FOLDERS=true
INGENIOUS_FILE_STORAGE__REVISIONS__URL=https://myaccount.blob.core.windows.net/prompts/
INGENIOUS_FILE_STORAGE__REVISIONS__TOKEN=${AZURE_STORAGE_CONNECTION_STRING}
INGENIOUS_FILE_STORAGE__REVISIONS__AUTHENTICATION_METHOD=token

# Data (e.g., uploaded datasets)
INGENIOUS_FILE_STORAGE__DATA__ENABLE=true
INGENIOUS_FILE_STORAGE__DATA__STORAGE_TYPE=azure
INGENIOUS_FILE_STORAGE__DATA__CONTAINER_NAME=data
INGENIOUS_FILE_STORAGE__DATA__PATH=ingenious-files
INGENIOUS_FILE_STORAGE__DATA__ADD_SUB_FOLDERS=true
INGENIOUS_FILE_STORAGE__DATA__URL=https://myaccount.blob.core.windows.net/data/
INGENIOUS_FILE_STORAGE__DATA__TOKEN=${AZURE_STORAGE_CONNECTION_STRING}
INGENIOUS_FILE_STORAGE__DATA__AUTHENTICATION_METHOD=token
```

> **Tip:** Do not quote the values with `"..."` in `.env` unless the value itself contains spaces or special characters that require quoting.

---

### 3) Validate Configuration
```bash
uv run ingen validate  # Check configuration before starting
```

**If validation fails with port conflicts:**
```bash
# Try a different port
INGENIOUS_WEB_CONFIGURATION__PORT=8001 uv run ingen validate

# Or make it permanent in .env
echo "INGENIOUS_WEB_CONFIGURATION__PORT=8001" >> .env
uv run ingen validate
```

> **⚠️ BREAKING CHANGE**: Ingenious now uses **pydantic-settings** for configuration via environment variables.
> Legacy YAML configuration files (`config.yml`, `profiles.yml`) are **no longer supported** and must be migrated to environment variables with `INGENIOUS_` prefixes. Use the migration script:
> ```bash
> uv run python scripts/migrate_config.py --yaml-file config.yml --output .env
> uv run python scripts/migrate_config.py --yaml-file profiles.yml --output .env.profiles
> ```

---

### 4) Start the Server
```bash
# Start server on port 8000 (recommended for development)
uv run ingen serve --port 8000

# Additional options:
# --host 0.0.0.0         # Bind host (default: 0.0.0.0)
# --port                 # Port to bind (default: 80 or $WEB_PORT env var)
# --config config.yml    # Legacy config file (deprecated — use environment variables)
# --profile production   # Legacy profile (deprecated — use environment variables)
```

---

### 5) Verify Health
```bash
# Check server health
curl http://localhost:8000/api/v1/health
```

---

## Test the Core Workflows

Create test files to avoid shell-escaping issues:

```bash
# Classification
cat > test_classification.json <<'JSON'
{
  "user_prompt": "Analyze this customer feedback: Great product",
  "conversation_flow": "classification-agent"
}
JSON

# Knowledge base (minimal)
cat > test_knowledge.json <<'JSON'
{
  "user_prompt": "Search for documentation about setup",
  "conversation_flow": "knowledge-base-agent"
}
JSON

# Knowledge base (with top_k control)
cat > test_knowledge_topk.json <<'JSON'
{
  "user_prompt": "Search for documentation about setup",
  "conversation_flow": "knowledge-base-agent",
  "kb_top_k": 3
}
JSON

# SQL manipulation (uses local SQLite by default unless you configured Azure SQL)
cat > test_sql.json <<'JSON'
{
  "user_prompt": "Show me all tables in the database",
  "conversation_flow": "sql-manipulation-agent"
}
JSON

# Run tests
curl -sS -X POST http://localhost:8000/api/v1/chat -H "Content-Type: application/json" -d @test_classification.json
curl -sS -X POST http://localhost:8000/api/v1/chat -H "Content-Type: application/json" -d @test_knowledge.json
curl -sS -X POST http://localhost:8000/api/v1/chat -H "Content-Type: application/json" -d @test_knowledge_topk.json
curl -sS -X POST http://localhost:8000/api/v1/chat -H "Content-Type: application/json" -d @test_sql.json
```

**Expected Responses**:
- **classification-agent**: JSON with message analysis and categories
- **knowledge-base-agent**: JSON with relevant info retrieved from **Azure AI Search** (or a clear message if your index is empty/misconfigured). For local-only mode (`KB_POLICY=local_only`), results come from the local store (make sure you’ve loaded documents).
- **sql-manipulation-agent**: JSON with query results or confirmation

**Common KB Misconfigurations**:
- `PreflightError: [azure_search] policy: Azure Search is required…`
  → Ensure `KB_USE_AZURE_SEARCH=true` **and** the `INGENIOUS_AZURE_SEARCH_SERVICES__0__...` block is present.
- 404/401/403 from Azure Search GET calls
  → Check `INDEX_NAME`, `ENDPOINT`, and `KEY`.

---

## Test the Template Workflow: `bike-insights`

> The `bike-insights` workflow is part of the project template generated by `uv run ingen init`.

**Recommended (use `parameters` field to avoid JSON-in-JSON quoting):**
```bash
cat > test_bike_insights.json <<'JSON'
{
  "user_prompt": "Analyze these store sales and summarize insights.",
  "conversation_flow": "bike-insights",
  "parameters": {
    "revision_id": "test-v1",
    "identifier": "test-001",
    "stores": [
      {
        "name": "Test Store",
        "location": "NSW",
        "bike_sales": [
          {
            "product_code": "MB-TREK-2021-XC",
            "quantity_sold": 2,
            "sale_date": "2023-04-01",
            "year": 2023,
            "month": "April",
            "customer_review": { "rating": 4.5, "comment": "Great bike" }
          }
        ],
        "bike_stock": []
      }
    ]
  }
}
JSON

curl -sS -X POST http://localhost:8000/api/v1/chat -H "Content-Type: application/json" -d @test_bike_insights.json
```

---

## Workflow Categories

### Core Library Workflows (Always Available)
- `classification-agent` — Text classification and routing
- `knowledge-base-agent` — Knowledge retrieval (**defaults to Azure AI Search**). Local ChromaDB supported with `KB_POLICY=local_only`/`prefer_local` (install `chromadb`).
- `sql-manipulation-agent` — Execute SQL queries from natural language (uses local SQLite by default unless Azure SQL is configured)

> Both hyphenated (`classification-agent`) and underscored (`classification_agent`) names are supported.

### Template Workflows (Created by `ingen init`)
- `bike-insights` — Multi-agent example for sales analysis (**only after `ingen init`**)

---

## Troubleshooting

- See the [detailed troubleshooting guide](docs/getting-started/troubleshooting.md) for port conflicts, configuration errors, and workflow issues.
- **Azure AI Search sanity check** (replace placeholders):
  ```bash
  curl -sD- -H "api-key: <your-search-key>"     "https://<your-service>.search.windows.net/indexes/<your-index>?api-version=2023-11-01"
  # Expect HTTP/1.1 200 OK
  ```

---

## Security Notes

- Do **NOT** commit `.env` files to source control.
- Redact secrets in logs. Avoid printing `api_key`, `key`, `Authorization`, `password`, `token`, etc.
- Rotate keys if accidentally exposed.

---

## Documentation

For detailed documentation, see the [docs](https://insight-services-apac.github.io/ingenious/).

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](https://github.com/Insight-Services-APAC/ingenious/blob/main/CONTRIBUTING.md).

## License

This project is licensed under the terms specified in the [LICENSE](https://github.com/Insight-Services-APAC/ingenious/blob/main/LICENSE) file.
