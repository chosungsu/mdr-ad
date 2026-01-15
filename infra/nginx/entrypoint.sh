#!/usr/bin/env sh
set -e

echo "[entrypoint] Starting BistelligencePlatform..."

# Set PORT with fallback to 80 for local environments
: "${PORT:=80}"
export PORT

echo "[entrypoint] Target PORT: $PORT"

# Validate PORT is numeric
if ! echo "$PORT" | grep -Eq '^[0-9]+$'; then
  echo "[entrypoint] ERROR: PORT must be numeric, got: $PORT"
  exit 1
fi

# Validate PORT range (1-65535)
if [ "$PORT" -lt 1 ] || [ "$PORT" -gt 65535 ]; then
  echo "[entrypoint] ERROR: PORT must be between 1-65535, got: $PORT"
  exit 1
fi

echo "[entrypoint] Rendering Nginx configuration from template..."
# Substitute only PORT variable in template
envsubst '${PORT}' < /etc/nginx/conf.d/default.conf.template > /etc/nginx/conf.d/default.conf

# Verify template was rendered
if [ ! -f /etc/nginx/conf.d/default.conf ]; then
  echo "[entrypoint] ERROR: Failed to create /etc/nginx/conf.d/default.conf"
  exit 1
fi

echo "[entrypoint] Generated Nginx config:"
cat /etc/nginx/conf.d/default.conf

echo "[entrypoint] Validating Nginx configuration..."
# Test nginx config before starting
if ! nginx -t; then
  echo "[entrypoint] ERROR: Nginx configuration validation failed"
  exit 1
fi

echo "[entrypoint] Nginx configuration valid"
echo "[entrypoint] Starting Supervisor (Backend + Nginx)..."

# Start supervisor to manage both processes
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf
