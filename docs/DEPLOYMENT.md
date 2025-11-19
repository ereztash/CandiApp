# CandiApp Deployment Guide

This guide covers deploying CandiApp to production environments.

---

## Quick Start (Development)

```bash
# 1. Clone repository
git clone https://github.com/your-org/candiapp.git
cd candiapp

# 2. Start services
docker-compose up -d

# 3. Access application
open http://localhost:3000
```

---

## Production Deployment

### Prerequisites

- Docker & Docker Compose
- Domain name with DNS configured
- SSL certificate (Let's Encrypt recommended)

### Step 1: Environment Configuration

Create `.env` file in project root:

```bash
# Database
POSTGRES_PASSWORD=your-secure-password-here

# Authentication
SECRET_KEY=your-super-secret-key-change-this

# Optional
ENVIRONMENT=production
LOG_LEVEL=INFO
```

### Step 2: SSL Certificates

Option A: Let's Encrypt (Recommended)
```bash
# Install certbot
sudo apt install certbot

# Generate certificate
sudo certbot certonly --standalone -d yourdomain.com

# Copy certificates
mkdir -p nginx/ssl
sudo cp /etc/letsencrypt/live/yourdomain.com/fullchain.pem nginx/ssl/
sudo cp /etc/letsencrypt/live/yourdomain.com/privkey.pem nginx/ssl/
```

Option B: Self-signed (Development only)
```bash
mkdir -p nginx/ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout nginx/ssl/privkey.pem \
  -out nginx/ssl/fullchain.pem \
  -subj "/CN=localhost"
```

### Step 3: Deploy

```bash
# Build and start services
docker-compose -f docker-compose.prod.yml up -d --build

# Check status
docker-compose -f docker-compose.prod.yml ps

# View logs
docker-compose -f docker-compose.prod.yml logs -f
```

### Step 4: Database Migration

```bash
# Run migrations
docker-compose -f docker-compose.prod.yml exec api alembic upgrade head
```

### Step 5: Verify Deployment

```bash
# Check health
curl https://yourdomain.com/api/v1/health

# Check frontend
curl https://yourdomain.com/
```

---

## Architecture

```
                    ┌─────────────┐
                    │   Nginx     │
                    │   (443)     │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
        ┌─────▼─────┐ ┌────▼────┐ ┌─────▼─────┐
        │ Frontend  │ │   API   │ │  Static   │
        │  (React)  │ │(FastAPI)│ │  Assets   │
        └───────────┘ └────┬────┘ └───────────┘
                           │
              ┌────────────┼────────────┐
              │                         │
        ┌─────▼─────┐           ┌───────▼───────┐
        │PostgreSQL │           │     Redis     │
        │   (5432)  │           │    (6379)     │
        └───────────┘           └───────────────┘
```

---

## Service Ports

| Service    | Internal | External |
|------------|----------|----------|
| Frontend   | 80       | -        |
| API        | 8000     | -        |
| Nginx      | 80, 443  | 80, 443  |
| PostgreSQL | 5432     | -        |
| Redis      | 6379     | -        |

---

## Environment Variables

### Required

| Variable          | Description                    |
|-------------------|--------------------------------|
| POSTGRES_PASSWORD | Database password              |
| SECRET_KEY        | JWT signing key                |

### Optional

| Variable                    | Default     | Description          |
|-----------------------------|-------------|----------------------|
| ENVIRONMENT                 | production  | Environment name     |
| LOG_LEVEL                   | INFO        | Logging level        |
| LOG_FORMAT                  | json        | Log format           |
| ACCESS_TOKEN_EXPIRE_MINUTES | 30          | Token expiration     |
| MAX_UPLOAD_SIZE_MB          | 10          | Max file upload size |

---

## Scaling

### Horizontal Scaling

```yaml
# docker-compose.prod.yml
services:
  api:
    deploy:
      replicas: 3
```

### Vertical Scaling

Adjust resource limits:

```yaml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
```

---

## Monitoring

### Health Checks

All services expose health endpoints:

- **API**: `GET /api/v1/health`
- **Frontend**: `GET /health`
- **Nginx**: `GET /health`

### Logs

```bash
# All services
docker-compose -f docker-compose.prod.yml logs -f

# Specific service
docker-compose -f docker-compose.prod.yml logs -f api

# JSON logs (production)
docker-compose -f docker-compose.prod.yml logs api | jq .
```

### Metrics

Prometheus metrics available at `/metrics` (if enabled).

---

## Backup & Recovery

### Database Backup

```bash
# Create backup
docker-compose -f docker-compose.prod.yml exec postgres \
  pg_dump -U candiapp candiapp > backup.sql

# Restore backup
docker-compose -f docker-compose.prod.yml exec -T postgres \
  psql -U candiapp candiapp < backup.sql
```

### Automated Backups

Add to crontab:
```bash
0 2 * * * /path/to/backup-script.sh
```

---

## Troubleshooting

### Common Issues

#### Database Connection Failed
```bash
# Check database is running
docker-compose -f docker-compose.prod.yml ps postgres

# Check logs
docker-compose -f docker-compose.prod.yml logs postgres
```

#### API Not Responding
```bash
# Check API health
curl http://localhost:8000/api/v1/health

# Check logs
docker-compose -f docker-compose.prod.yml logs api
```

#### SSL Certificate Issues
```bash
# Verify certificate
openssl s_client -connect yourdomain.com:443 -servername yourdomain.com

# Check certificate files
ls -la nginx/ssl/
```

### Reset Everything

```bash
# Stop and remove all
docker-compose -f docker-compose.prod.yml down -v

# Remove images
docker-compose -f docker-compose.prod.yml down --rmi all

# Fresh start
docker-compose -f docker-compose.prod.yml up -d --build
```

---

## Security Checklist

- [ ] Strong POSTGRES_PASSWORD (32+ characters)
- [ ] Unique SECRET_KEY for JWT
- [ ] Valid SSL certificate
- [ ] Firewall configured (only 80/443 exposed)
- [ ] Rate limiting enabled
- [ ] CORS properly configured
- [ ] Database not exposed externally
- [ ] Regular backups configured
- [ ] Log monitoring in place

---

## Cloud Deployment

### AWS

```bash
# Using ECS
aws ecs create-cluster --cluster-name candiapp
# ... deploy task definitions
```

### Google Cloud

```bash
# Using Cloud Run
gcloud run deploy candiapp-api --source .
```

### DigitalOcean

```bash
# Using App Platform
doctl apps create --spec .do/app.yaml
```

---

## Support

- Documentation: `docs/`
- Issues: GitHub Issues
- API Reference: `/api/v1/docs`

---

*Last Updated: 2025-01-01*
