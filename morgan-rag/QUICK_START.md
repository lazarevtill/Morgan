# Morgan RAG Quick Start

## Setup and Start

1. **Create .env file** (if not exists):
   ```powershell
   python create_env_file.py
   ```

2. **Start all services**:
   ```powershell
   docker compose up -d --build
   ```

   Or use the batch script:
   ```cmd
   start_morgan.bat
   ```

3. **Wait for services to start** (about 45 seconds)

4. **Test the setup**:
   ```powershell
   python test_setup.py
   ```

## Verify Services

- **Web Interface**: http://localhost:8080
- **API**: http://localhost:8000
- **Qdrant**: http://localhost:6333
- **Redis**: localhost:6379

## Check Logs

```powershell
# All services
docker compose logs -f

# Just Morgan
docker compose logs -f morgan

# Just Qdrant
docker compose logs -f qdrant
```

## Stop Services

```powershell
docker compose down
```

## Configuration

All configuration is in `.env` file. Key settings:

- `LLM_BASE_URL`: Your LLM endpoint (https://ai.ishosting.com/api)
- `LLM_API_KEY`: Your API key
- `LLM_MODEL`: Model to use (gemma3:latest)
- `OLLAMA_HOST`: Ollama host for embeddings (192.168.100.88:11434)
- `EMBEDDING_MODEL`: Embedding model (qwen3-embedding:latest)

## Troubleshooting

1. **Services won't start**: Check logs with `docker compose logs`
2. **LLM connection fails**: Verify `LLM_BASE_URL` and `LLM_API_KEY` in `.env`
3. **Embedding service fails**: Check `OLLAMA_HOST` is reachable from container
4. **Qdrant connection fails**: Ensure Qdrant container is running


