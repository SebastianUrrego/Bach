#!/bin/bash
# Arranca el servidor MLflow. Lo llama bach-mlflow.service con las variables de /etc/bach/bach.env.
# MLflow 3 solo acepta por defecto el Host localhost o IPs privadas, así que en cada arranque se
# agrega la IP pública (Elastic IP) que reporta el servicio de metadatos de la EC2.
set -euo pipefail

imds() {
  local token
  token=$(curl -sf -X PUT http://169.254.169.254/latest/api/token \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 60") || return 0
  curl -sf -H "X-aws-ec2-metadata-token: $token" "http://169.254.169.254/latest/meta-data/$1" || true
}

PUBLIC_IP=$(imds public-ipv4)
PRIVATE_IP=$(imds local-ipv4)

HOSTS="localhost,localhost:${MLFLOW_PORT},127.0.0.1,127.0.0.1:${MLFLOW_PORT}"
ORIGINS="http://localhost:${MLFLOW_PORT},http://127.0.0.1:${MLFLOW_PORT}"
for ip in $PUBLIC_IP $PRIVATE_IP; do
  HOSTS="$HOSTS,$ip,$ip:${MLFLOW_PORT}"
  ORIGINS="$ORIGINS,http://$ip:${MLFLOW_PORT}"
done
if [ -n "${EXTRA_ALLOWED_HOSTS:-}" ]; then
  HOSTS="$HOSTS,$EXTRA_ALLOWED_HOSTS"
fi

echo "MLflow: IP pública=${PUBLIC_IP:-ninguna} artefactos=${ARTIFACTS_DESTINATION}"
exec /opt/bach/.venv/bin/mlflow server \
  --host 0.0.0.0 --port "${MLFLOW_PORT}" --workers "${MLFLOW_WORKERS:-2}" \
  --backend-store-uri "sqlite:////var/lib/bach/mlflow.db" \
  --serve-artifacts --artifacts-destination "${ARTIFACTS_DESTINATION}" \
  --allowed-hosts "$HOSTS" --cors-allowed-origins "$ORIGINS"
