# Stage 1: Build frontend assets (Vite)
FROM node:20-alpine AS frontend-build

WORKDIR /app

COPY package*.json ./
COPY vite.config.ts ./
COPY tsconfig.json ./
RUN npm ci

# Copy only frontend source (avoid pulling backend/large folders)
COPY index.html ./
COPY index.tsx ./
COPY App.tsx ./
COPY types.ts ./
COPY metadata.json ./
COPY components ./components
COPY utils ./utils

# Build-time env for frontend API base (same-origin via nginx /api)
# Production: uses /api (proxied by nginx)
# Local dev: override with VITE_API_BASE=http://localhost:8000
ENV VITE_API_BASE=/api

RUN npm run build


# Stage 2: Final stage - Nginx + Backend (supervisor)
# Use Debian-based nginx image for PyTorch wheels compatibility (avoid Alpine/musl).
FROM nginx:1.27

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
     python3 \
     python3-pip \
     supervisor \
     ca-certificates \
     gettext-base \
  && rm -rf /var/lib/apt/lists/*

# Install backend dependencies
WORKDIR /app/packages/backend
COPY packages/backend/requirements.txt ./
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

# Copy backend source
COPY packages/backend/ ./

# Copy MDRAD python definition files needed for GCS-loaded model inference
RUN mkdir -p /app/modeling/mdrad
COPY modeling/mdrad/*.py /app/modeling/mdrad/

# Set environment variable for production (behind Nginx proxy)
ENV USE_API_PREFIX=true

# Copy built frontend assets into nginx html directory
COPY --from=frontend-build /app/dist /usr/share/nginx/html

# Nginx configuration (static, port 80)
COPY infra/nginx/nginx.conf /etc/nginx/conf.d/default.conf

# Supervisor configuration: run backend + nginx in one container
COPY infra/supervisor/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Expose port 80 for nginx
EXPOSE 80

# Start supervisor directly (no entrypoint needed)
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
