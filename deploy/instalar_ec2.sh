#!/bin/bash
# Instala MLflow y la API del laboratorio Bach como servicios systemd en una EC2 Ubuntu 24.04.
#
# Uso (desde Session Manager, con el repo clonado en /opt/bach):
#   sudo BUCKET=mi-bucket bash /opt/bach/deploy/instalar_ec2.sh     # artefactos en S3
#   sudo bash /opt/bach/deploy/instalar_ec2.sh                      # artefactos en el disco de la EC2
#
# Se puede volver a ejecutar para actualizar: reinstala dependencias y reinicia los servicios,
# pero no sobrescribe /etc/bach/bach.env si ya existe.
set -euo pipefail

REPO=/opt/bach
DATA=/var/lib/bach
CONF=/etc/bach/bach.env

if [ "$(id -u)" -ne 0 ]; then
  echo "Ejecutar con sudo" >&2
  exit 1
fi
if [ ! -f "$REPO/pyproject.toml" ]; then
  echo "No encuentro el repo en $REPO (git clone https://github.com/SebastianUrrego/Bach.git $REPO)" >&2
  exit 1
fi

echo "==> Paquetes del sistema"
apt-get update -q
apt-get install -y -q python3-venv python3-pip git curl

echo "==> Usuario y carpetas"
id bach >/dev/null 2>&1 || useradd --system --home "$DATA" --shell /usr/sbin/nologin bach
mkdir -p "$DATA" /etc/bach
chown bach:bach "$DATA"

echo "==> Entorno de Python (torch CPU, MLflow, FastAPI)"
[ -d "$REPO/.venv" ] || python3 -m venv "$REPO/.venv"
"$REPO/.venv/bin/pip" install -q --upgrade pip
"$REPO/.venv/bin/pip" install -q -r "$REPO/requirements.txt" boto3
"$REPO/.venv/bin/pip" install -q -e "$REPO"

echo "==> Configuración"
if [ ! -f "$CONF" ]; then
  if [ -n "${BUCKET:-}" ]; then
    ARTIFACTS="s3://${BUCKET}/mlflow"
  else
    ARTIFACTS="$DATA/mlartifacts"
  fi
  cat > "$CONF" <<CONFEOF
# Configuración de los servicios bach-mlflow y bach-api
MLFLOW_PORT=5000
API_PORT=8000
MLFLOW_WORKERS=2
ARTIFACTS_DESTINATION=${ARTIFACTS}
AWS_DEFAULT_REGION=us-east-1
# Hosts extra permitidos en MLflow (por ejemplo un nombre DNS), separados por comas
EXTRA_ALLOWED_HOSTS=
# La API carga este modelo desde el servidor MLflow local
MLFLOW_TRACKING_URI=http://127.0.0.1:5000
MODEL_NAME=bach-chorales-equipoXX
MODEL_ALIAS=champion
CONFEOF
  echo "Creado $CONF (revisar MODEL_NAME)"
else
  echo "Se conserva $CONF"
fi

echo "==> Servicios"
cp "$REPO/deploy/bach-mlflow.service" "$REPO/deploy/bach-api.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable bach-mlflow bach-api
systemctl restart bach-mlflow
systemctl restart bach-api

echo "==> Esperando a MLflow"
for i in $(seq 60); do
  curl -sf http://127.0.0.1:5000/health >/dev/null && break
  sleep 2
done
systemctl --no-pager --lines=3 status bach-mlflow bach-api || true

TOKEN=$(curl -sf -X PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 60" || true)
IP=$(curl -sf -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/public-ipv4 || echo "<IP>")
echo
echo "MLflow: http://$IP:5000"
echo "API:    http://$IP:8000/docs  (responde 503 en /health hasta registrar el modelo con alias champion)"
