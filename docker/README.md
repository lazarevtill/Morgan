# Docker Deployment

This directory contains Docker configurations for deploying Morgan.

## Quick Start

1. **Start all services:**

```bash
docker-compose up -d
```

2. **Pull the LLM model:**

```bash
docker exec -it morgan-ollama ollama pull gemma3
```

3. **Check health:**

```bash
curl http://localhost:8080/health
```

4. **View logs:**

```bash
docker-compose logs -f morgan-server
```

## Services

- **morgan-server** (port 8080): Main Morgan server
- **qdrant** (port 6333): Vector database
- **ollama** (port 11434): LLM service
- **prometheus** (port 9090): Metrics (optional, use `--profile monitoring`)

## Configuration

Edit `docker-compose.yml` to customize:
- Environment variables
- Port mappings
- Volume mounts
- Resource limits

## Monitoring

To enable Prometheus monitoring:

```bash
docker-compose --profile monitoring up -d
```

Access Prometheus at http://localhost:9090

## Stopping

```bash
docker-compose down
```

To remove volumes:

```bash
docker-compose down -v
```

## Building

To rebuild the server image:

```bash
docker-compose build morgan-server
```

## Troubleshooting

**Server won't start:**
- Check logs: `docker-compose logs morgan-server`
- Verify Qdrant is running: `curl http://localhost:6333`
- Verify Ollama is running: `curl http://localhost:11434`

**Out of memory:**
- Reduce embedding model size in environment variables
- Add memory limits to docker-compose.yml

**Slow responses:**
- Ensure Ollama has pulled the model
- Check if GPU is available for Ollama
- Monitor resource usage: `docker stats`
