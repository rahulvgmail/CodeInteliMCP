# 🚢 Deployment Guide

## Overview

This guide covers deployment strategies for the Code Intelligence MCP server in various environments.

## 1. Local Development Setup

### Quick Start

```bash
# Clone repository
git clone <repository_url>
cd treeSitterMCP

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Initialize configuration
export CODE_INTEL_PROJECT_ROOT="$PWD"
export CODE_INTEL_DATA_DIR="$PWD/data"

# Run server
python -m code_intelligence_mcp.server
```

### Development Configuration

Create a `.env` file for local development:

```env
# .env
CODE_INTEL_PROJECT_ROOT=/path/to/your/project
CODE_INTEL_DATA_DIR=/path/to/data/storage
CODE_INTEL_LOG_LEVEL=DEBUG
CODE_INTEL_MAX_WORKERS=4
```

### Docker Development

```dockerfile
# Dockerfile.dev
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements
COPY pyproject.toml .
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY code_intelligence_mcp/ code_intelligence_mcp/

# Create data directory
RUN mkdir -p /data

# Set environment variables
ENV CODE_INTEL_DATA_DIR=/data
ENV PYTHONPATH=/app

# Expose MCP port
EXPOSE 5000

# Run server
CMD ["python", "-m", "code_intelligence_mcp.server"]
```

Build and run:

```bash
docker build -f Dockerfile.dev -t code-intel-mcp:dev .
docker run -it \
  -v $(pwd):/project:ro \
  -v $(pwd)/data:/data \
  -e CODE_INTEL_PROJECT_ROOT=/project \
  code-intel-mcp:dev
```

## 2. Production Deployment

### System Requirements

- **CPU**: 4+ cores recommended
- **RAM**: 8GB minimum, 16GB+ for large codebases
- **Storage**: SSD with 10GB+ free space
- **OS**: Linux (Ubuntu 20.04+), macOS, or Windows with WSL2

### Production Configuration

```python
# code_intelligence_mcp/config/production.py
from ..config import Config, GraphConfig, VectorConfig, IndexingConfig


class ProductionConfig(Config):
    """Production configuration."""
    
    def __init__(self):
        super().__init__()
        
        # Performance settings
        self.max_workers = 8
        
        # Graph settings
        self.graph = GraphConfig(
            cache_size_mb=512,
            connection_timeout=10000,
            enable_progress_bar=False
        )
        
        # Vector settings
        self.vector = VectorConfig(
            embedding_batch_size=64,
            similarity_threshold=0.8
        )
        
        # Indexing settings
        self.indexing = IndexingConfig(
            incremental_batch_size=50,
            full_index_file_limit=50000,
            file_size_limit_mb=20.0,
            auto_index_threshold=10
        )
        
        # Security settings
        self.enable_auth = True
        self.api_key_header = "X-API-Key"
        self.allowed_origins = ["*"]
        
        # Monitoring
        self.enable_metrics = True
        self.metrics_port = 9090
```

### Systemd Service

Create a systemd service for automatic startup:

```ini
# /etc/systemd/system/code-intel-mcp.service
[Unit]
Description=Code Intelligence MCP Server
After=network.target

[Service]
Type=simple
User=mcp-user
Group=mcp-group
WorkingDirectory=/opt/code-intel-mcp
Environment="PATH=/opt/code-intel-mcp/venv/bin"
Environment="CODE_INTEL_PROJECT_ROOT=/var/projects"
Environment="CODE_INTEL_DATA_DIR=/var/lib/code-intel-mcp"
ExecStart=/opt/code-intel-mcp/venv/bin/python -m code_intelligence_mcp.server
Restart=always
RestartSec=10

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/code-intel-mcp /var/projects

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable code-intel-mcp
sudo systemctl start code-intel-mcp
```

### Docker Production

```dockerfile
# Dockerfile
FROM python:3.9-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production image
FROM python:3.9-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user
RUN groupadd -r mcp && useradd -r -g mcp mcp

# Create directories
RUN mkdir -p /app /data && chown -R mcp:mcp /app /data

# Copy application
WORKDIR /app
COPY --chown=mcp:mcp code_intelligence_mcp/ code_intelligence_mcp/

# Switch to non-root user
USER mcp

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:5000/health')"

# Environment
ENV CODE_INTEL_DATA_DIR=/data
ENV PYTHONPATH=/app

# Expose port
EXPOSE 5000

# Run server
CMD ["python", "-m", "code_intelligence_mcp.server"]
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  code-intel-mcp:
    build: .
    container_name: code-intel-mcp
    restart: unless-stopped
    ports:
      - "5000:5000"
      - "9090:9090"  # Metrics
    volumes:
      - ./data:/data
      - ${PROJECT_PATH}:/project:ro
    environment:
      - CODE_INTEL_PROJECT_ROOT=/project
      - CODE_INTEL_DATA_DIR=/data
      - CODE_INTEL_LOG_LEVEL=INFO
    networks:
      - mcp-network
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '2'
          memory: 4G

  redis:
    image: redis:7-alpine
    container_name: code-intel-redis
    restart: unless-stopped
    volumes:
      - redis-data:/data
    networks:
      - mcp-network

  prometheus:
    image: prom/prometheus
    container_name: code-intel-prometheus
    restart: unless-stopped
    ports:
      - "9091:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    networks:
      - mcp-network

volumes:
  redis-data:
  prometheus-data:

networks:
  mcp-network:
    driver: bridge
```

## 3. Kubernetes Deployment

### Kubernetes Manifests

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: code-intel-mcp
  labels:
    app: code-intel-mcp
spec:
  replicas: 3
  selector:
    matchLabels:
      app: code-intel-mcp
  template:
    metadata:
      labels:
        app: code-intel-mcp
    spec:
      containers:
      - name: mcp-server
        image: code-intel-mcp:latest
        ports:
        - containerPort: 5000
          name: mcp
        - containerPort: 9090
          name: metrics
        env:
        - name: CODE_INTEL_PROJECT_ROOT
          value: /project
        - name: CODE_INTEL_DATA_DIR
          value: /data
        - name: REDIS_URL
          value: redis://redis-service:6379
        volumeMounts:
        - name: data
          mountPath: /data
        - name: project
          mountPath: /project
          readOnly: true
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        livenessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 5000
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: code-intel-data-pvc
      - name: project
        configMap:
          name: project-files

---
apiVersion: v1
kind: Service
metadata:
  name: code-intel-mcp-service
spec:
  selector:
    app: code-intel-mcp
  ports:
  - name: mcp
    port: 5000
    targetPort: 5000
  - name: metrics
    port: 9090
    targetPort: 9090
  type: ClusterIP

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: code-intel-data-pvc
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 50Gi
  storageClassName: fast-ssd
```

### Helm Chart

```yaml
# helm/code-intel-mcp/values.yaml
replicaCount: 3

image:
  repository: code-intel-mcp
  pullPolicy: IfNotPresent
  tag: "latest"

service:
  type: ClusterIP
  port: 5000
  metricsPort: 9090

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: code-intel.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: code-intel-tls
      hosts:
        - code-intel.example.com

resources:
  limits:
    cpu: 4
    memory: 8Gi
  requests:
    cpu: 2
    memory: 4Gi

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

persistence:
  enabled: true
  storageClass: fast-ssd
  accessMode: ReadWriteOnce
  size: 50Gi

redis:
  enabled: true
  architecture: standalone
  auth:
    enabled: false

monitoring:
  enabled: true
  serviceMonitor:
    enabled: true
```

## 4. Integration with Claude Code

### Claude Code Configuration

```json
{
  "mcpServers": {
    "code-intelligence": {
      "command": "docker",
      "args": [
        "run",
        "--rm",
        "-v", "${PROJECT_ROOT}:/project:ro",
        "-v", "${HOME}/.code-intel/data:/data",
        "-e", "CODE_INTEL_PROJECT_ROOT=/project",
        "code-intel-mcp:latest"
      ],
      "env": {
        "PROJECT_ROOT": "${workspaceFolder}"
      }
    }
  }
}
```

### Remote MCP Server

For production deployments, use remote MCP:

```json
{
  "mcpServers": {
    "code-intelligence": {
      "type": "http",
      "url": "https://code-intel.example.com/mcp",
      "headers": {
        "Authorization": "Bearer ${CODE_INTEL_API_KEY}"
      }
    }
  }
}
```

## 5. Monitoring and Operations

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'code-intel-mcp'
    static_configs:
      - targets: ['code-intel-mcp:9090']
    metrics_path: '/metrics'
```

### Grafana Dashboard

Create dashboards for:
- Query performance (p50, p95, p99)
- Indexing rate and progress
- Cache hit rates
- Memory and CPU usage
- Error rates

### Logging

Configure structured logging:

```python
# code_intelligence_mcp/utils/logging.py
import logging
import json
from pythonjsonlogger import jsonlogger


def setup_logging(level: str = "INFO"):
    """Setup structured JSON logging."""
    logHandler = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter(
        fmt='%(asctime)s %(name)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logHandler.setFormatter(formatter)
    
    logging.root.setLevel(level)
    logging.root.addHandler(logHandler)
```

### Backup and Recovery

```bash
#!/bin/bash
# backup.sh

# Backup DuckDB database
docker exec code-intel-mcp \
  duckdb /data/code_graph.duckdb \
  ".backup /data/backup_$(date +%Y%m%d).db"

# Backup ChromaDB
docker exec code-intel-mcp \
  tar -czf /data/chroma_backup_$(date +%Y%m%d).tar.gz \
  /data/chroma

# Upload to S3
aws s3 sync /data/backups s3://my-bucket/code-intel-backups/
```

## 6. Security Considerations

### API Authentication

```python
# code_intelligence_mcp/auth/api_key.py
from functools import wraps
from flask import request, jsonify


def require_api_key(f):
    """Require API key for endpoint."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        
        if not api_key or not validate_api_key(api_key):
            return jsonify({'error': 'Invalid API key'}), 401
        
        return f(*args, **kwargs)
    
    return decorated_function
```

### Network Security

- Use TLS for all connections
- Implement rate limiting
- Use network policies in Kubernetes
- Regularly update dependencies

### Data Security

- Encrypt data at rest
- Use read-only mounts for source code
- Implement access controls
- Audit log all operations

## 7. Performance Tuning

### Database Tuning

```sql
-- Optimize DuckDB settings
PRAGMA memory_limit='4GB';
PRAGMA threads=8;
PRAGMA enable_profiling=false;
PRAGMA checkpoint_threshold='1GB';
```

### Resource Limits

Set appropriate resource limits based on codebase size:

| Codebase Size | CPU | Memory | Storage |
|---------------|-----|--------|---------|
| < 10k files   | 2   | 4GB    | 10GB    |
| 10k-50k files | 4   | 8GB    | 50GB    |
| 50k-100k files| 8   | 16GB   | 100GB   |
| > 100k files  | 16  | 32GB   | 200GB+  |

## 8. Troubleshooting

### Common Issues

1. **Out of Memory**
   - Increase memory limits
   - Enable swap
   - Reduce batch sizes

2. **Slow Indexing**
   - Check disk I/O
   - Increase worker count
   - Enable parallel processing

3. **Connection Issues**
   - Check firewall rules
   - Verify service discovery
   - Check TLS certificates

### Debug Mode

Enable debug mode for troubleshooting:

```bash
export CODE_INTEL_LOG_LEVEL=DEBUG
export CODE_INTEL_ENABLE_PROFILING=true
python -m code_intelligence_mcp.server --debug
```

This comprehensive deployment guide covers all aspects of deploying the Code Intelligence MCP server in various environments.
