# Docker Deployment Guide

Complete guide for deploying CandiApp using Docker and Docker Compose.

---

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/ereztash/CandiApp.git
cd CandiApp

# Build and run
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f candiapp
```

**That's it!** CandiApp is now running at `http://localhost:8000`

---

## 📦 What's Included

### Services

1. **candiapp** - Main application container
   - Port: 8000
   - Includes all NLP models
   - Auto-downloads spaCy models on build

2. **redis** - Caching layer
   - Port: 6379
   - Persistent data volume

3. **postgres** - Database (optional)
   - Port: 5432
   - Persistent data volume

4. **nginx** - Reverse proxy (optional)
   - Ports: 80, 443
   - SSL support

---

## 🔧 Configuration

### Environment Variables

Create `.env` file in project root:

```bash
# Application Settings
CANDIAPP_ENV=production
CANDIAPP_DEBUG=false
CANDIAPP_LOG_LEVEL=INFO

# Feature Flags
CANDIAPP_USE_GPU=false
CANDIAPP_USE_BERT=true
CANDIAPP_USE_SPACY_LARGE=true
CANDIAPP_SEMANTIC_MATCHING=true

# Multilingual
CANDIAPP_LANGUAGES=en,es,fr,de,he
CANDIAPP_DEFAULT_LANGUAGE=en

# Performance
CANDIAPP_WORKERS=4
CANDIAPP_TIMEOUT=300

# Database (if using)
CANDIAPP_DB_URL=postgresql://candiapp:candiapp_password@postgres:5432/candiapp

# Redis
CANDIAPP_REDIS_URL=redis://redis:6379/0

# Security
CANDIAPP_SECRET_KEY=your-secret-key-here
CANDIAPP_ALLOWED_HOSTS=localhost,127.0.0.1
```

### docker-compose.yml

Customize services as needed:

```yaml
version: '3.8'

services:
  candiapp:
    build: .
    image: candiapp:latest
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    env_file:
      - .env
    depends_on:
      - redis
      - postgres
```

---

## 🏗️ Building

### Standard Build

```bash
docker build -t candiapp:latest .
```

### With Build Args

```bash
docker build \
  --build-arg PYTHON_VERSION=3.11 \
  --build-arg SPACY_MODELS="en es fr de" \
  -t candiapp:latest .
```

### Multi-platform Build

```bash
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t candiapp:latest .
```

---

## 🚢 Deployment Scenarios

### Scenario 1: Development

```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  candiapp:
    build:
      context: .
      dockerfile: Dockerfile
    volumes:
      - ./src:/app/src:ro  # Mount source code
      - ./data:/app/data
    environment:
      - CANDIAPP_ENV=development
      - CANDIAPP_DEBUG=true
      - CANDIAPP_LOG_LEVEL=DEBUG
    command: uvicorn candiapp.api:app --reload --host 0.0.0.0
```

```bash
docker-compose -f docker-compose.dev.yml up
```

### Scenario 2: Production (Single Server)

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  candiapp:
    image: candiapp:latest
    restart: always
    environment:
      - CANDIAPP_ENV=production
      - CANDIAPP_WORKERS=4
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
```

```bash
docker-compose -f docker-compose.prod.yml up -d
```

### Scenario 3: GPU-Accelerated

```yaml
# docker-compose.gpu.yml
version: '3.8'

services:
  candiapp-gpu:
    build:
      context: .
      dockerfile: Dockerfile.gpu
    runtime: nvidia
    environment:
      - CANDIAPP_USE_GPU=true
      - NVIDIA_VISIBLE_DEVICES=all
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

Requirements:
- NVIDIA GPU
- nvidia-docker2 installed
- CUDA-enabled PyTorch

```bash
docker-compose -f docker-compose.gpu.yml up -d
```

### Scenario 4: Distributed/Scaled

```yaml
# docker-compose.scale.yml
version: '3.8'

services:
  candiapp:
    image: candiapp:latest
    deploy:
      replicas: 3
      restart_policy:
        condition: on-failure

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - candiapp
```

```bash
docker-compose -f docker-compose.scale.yml up -d --scale candiapp=3
```

---

## 🔒 Security

### Best Practices

1. **Use secrets** (not environment variables)

```yaml
services:
  candiapp:
    secrets:
      - db_password
      - secret_key

secrets:
  db_password:
    file: ./secrets/db_password.txt
  secret_key:
    file: ./secrets/secret_key.txt
```

2. **Run as non-root** (already configured)

```dockerfile
USER candiapp  # UID 1000
```

3. **Read-only filesystem** (where possible)

```yaml
services:
  candiapp:
    read_only: true
    tmpfs:
      - /tmp
      - /app/logs
```

4. **Network isolation**

```yaml
networks:
  frontend:
  backend:

services:
  nginx:
    networks:
      - frontend

  candiapp:
    networks:
      - frontend
      - backend

  postgres:
    networks:
      - backend  # Not exposed to frontend
```

5. **Use SSL/TLS**

```yaml
services:
  nginx:
    volumes:
      - ./ssl:/etc/nginx/ssl:ro
      - ./nginx/ssl.conf:/etc/nginx/conf.d/default.conf:ro
```

---

## 📊 Monitoring

### Health Checks

Built-in health check:

```yaml
services:
  candiapp:
    healthcheck:
      test: ["CMD", "python", "-c", "import candiapp; print('OK')"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s
```

### Prometheus Metrics

```yaml
services:
  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"
```

### Logging

```yaml
services:
  candiapp:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

With ELK stack:

```yaml
services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.11.0

  logstash:
    image: docker.elastic.co/logstash/logstash:8.11.0

  kibana:
    image: docker.elastic.co/kibana/kibana:8.11.0
    ports:
      - "5601:5601"
```

---

## 🔄 Updates & Maintenance

### Update Application

```bash
# Pull latest changes
git pull

# Rebuild image
docker-compose build candiapp

# Restart with new image
docker-compose up -d candiapp
```

### Update spaCy Models

```bash
# Access container
docker-compose exec candiapp bash

# Download new models
python -m spacy download en_core_web_lg

# Exit and restart
exit
docker-compose restart candiapp
```

### Backup Data

```bash
# Backup volumes
docker-compose exec postgres pg_dump -U candiapp candiapp > backup.sql

# Backup feature store
docker-compose exec candiapp tar -czf /tmp/features.tar.gz /app/data/features
docker cp candiapp:/tmp/features.tar.gz ./backup/
```

### Restore Data

```bash
# Restore database
docker-compose exec -T postgres psql -U candiapp < backup.sql

# Restore features
docker cp ./backup/features.tar.gz candiapp:/tmp/
docker-compose exec candiapp tar -xzf /tmp/features.tar.gz -C /
```

---

## 🎯 Performance Tuning

### CPU-Optimized

```yaml
services:
  candiapp:
    environment:
      - CANDIAPP_USE_GPU=false
      - CANDIAPP_USE_BERT=false  # Disable BERT for speed
      - CANDIAPP_WORKERS=8
    deploy:
      resources:
        limits:
          cpus: '4'
```

### GPU-Optimized

```yaml
services:
  candiapp-gpu:
    environment:
      - CANDIAPP_USE_GPU=true
      - CANDIAPP_USE_BERT=true
      - CANDIAPP_BATCH_SIZE=32
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
```

### Memory-Optimized

```yaml
services:
  candiapp:
    environment:
      - CANDIAPP_SPACY_MODEL=en_core_web_sm  # Use small models
      - CANDIAPP_CACHE_SIZE=100  # Limit cache
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G
```

---

## 🧪 Testing in Docker

```bash
# Run tests in container
docker-compose exec candiapp pytest

# Run specific test
docker-compose exec candiapp pytest tests/test_features.py

# Run with coverage
docker-compose exec candiapp pytest --cov=candiapp
```

---

## 🐛 Troubleshooting

### Container won't start

```bash
# Check logs
docker-compose logs candiapp

# Check health
docker-compose ps

# Rebuild from scratch
docker-compose down -v
docker-compose build --no-cache
docker-compose up -d
```

### Out of memory

```bash
# Increase Docker memory limit (Docker Desktop)
# Settings > Resources > Memory > 4GB+

# Or reduce resource usage
docker-compose exec candiapp \
  env CANDIAPP_SPACY_MODEL=en_core_web_sm \
  python -m candiapp
```

### spaCy model not found

```bash
# Download models
docker-compose exec candiapp python -m spacy download en_core_web_lg

# Or rebuild image
docker-compose build --no-cache candiapp
```

### Permission errors

```bash
# Fix volume permissions
sudo chown -R 1000:1000 ./data ./logs

# Or run as root (not recommended)
docker-compose exec -u root candiapp bash
```

---

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [CandiApp GitHub](https://github.com/ereztash/CandiApp)
- [Advanced Features Guide](./ADVANCED_FEATURES.md)

---

## 🆘 Getting Help

1. Check logs: `docker-compose logs -f`
2. Check documentation: `/docs`
3. GitHub Issues
4. Community Discord

---

**Version**: 2.0.0
**Last Updated**: 2024-11-18
