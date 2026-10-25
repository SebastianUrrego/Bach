# Despliegue en AWS (Learner Lab)

Una sola EC2 ejecuta dos servicios systemd:

| Servicio | Puerto | Qué hace |
|----------|--------|----------|
| `bach-mlflow` | 5000 | Servidor MLflow: experimentos, Model Registry y artefactos (en S3 o en el disco) |
| `bach-api` | 8000 | FastAPI: al arrancar carga `models:/<MODEL_NAME>@champion` desde el MLflow local |

```
PC (entrena E1–E4) ──HTTP──► MLflow EC2:5000 ──► S3 (artefactos, con LabRole)
API EC2:8000 ◄── champion desde MLflow (127.0.0.1:5000)
Docente ──► http://<ElasticIP>:5000   y   http://<ElasticIP>:8000/docs
```

Los dos servicios arrancan solos cada vez que se inicia el lab. El PC no necesita credenciales de AWS:
sube los artefactos a través del servidor MLflow (`--serve-artifacts`).

## 1. Crear los recursos (consola de AWS, región us-east-1)

1. **Bucket S3** (opcional): `bach-mlflow-artifacts-<algo-único>`, con las opciones por defecto (privado).
   Si no se crea, los artefactos quedan en `/var/lib/bach/mlartifacts` de la EC2.
2. **Security Group** `bach-sg`, en la VPC por defecto. Reglas de entrada:
   - TCP 5000, origen: tu IP (y la del docente).
   - TCP 8000, origen: tu IP (y la del docente).
   - No hace falta el puerto 22: la conexión es por Session Manager.
3. **EC2** `bach-mlflow-api`:
   - AMI Ubuntu Server 24.04 LTS (x86_64), tipo `t3.medium`.
   - Key pair: ninguno.
   - Red: subred pública con IP pública automática, Security Group `bach-sg`.
   - Almacenamiento: 20 GB gp3.
   - Detalles avanzados → Perfil de instancia IAM: `LabInstanceProfile`.
4. **Elastic IP**: asignarla y asociarla a `bach-mlflow-api`. Así la URL no cambia entre sesiones del lab.

## 2. Instalar (Session Manager)

EC2 → instancia → **Conectar** → **Session Manager**. Luego:

```bash
sudo apt-get update -q && sudo apt-get install -y -q git
sudo git clone https://github.com/SebastianUrrego/Bach.git /opt/bach
sudo BUCKET=bach-mlflow-artifacts-xxx bash /opt/bach/deploy/instalar_ec2.sh
```

- Sin bucket, omitir `BUCKET=...`.
- Si el repositorio es privado, clonar con un token de solo lectura:
  `https://<usuario>:<token>@github.com/...`.
- El script se ejecuta desde el clon en Linux, de modo que no hay problemas de fin de línea CRLF
  (el repo tiene `.gitattributes` con `eol=lf`).

Después, revisar el nombre del modelo en `/etc/bach/bach.env`:

```bash
sudo nano /etc/bach/bach.env      # MODEL_NAME=bach-chorales-equipoF
sudo systemctl restart bach-api
```

## 3. Verificar

- Desde el PC, abrir `http://<ElasticIP>:5000`: debe cargar la interfaz de MLflow.
- Hasta que se registre un modelo con alias `champion`, `http://<ElasticIP>:8000/health` responde
  503 con el motivo.
- Logs en la EC2:
  ```bash
  journalctl -u bach-mlflow -n 50 --no-pager
  journalctl -u bach-api -n 50 --no-pager
  ```

## 4. Usarlo desde el PC

En el `.env` local:

```
MLFLOW_TRACKING_URI=http://<ElasticIP>:5000
```

El entrenamiento y el registro (`log_model` + alias `champion`) se hacen desde el PC contra esa URL.
Después de asignar o cambiar `champion`, reiniciar la API para que cargue la nueva versión:

```bash
sudo systemctl restart bach-api
```

## 5. Actualizar el código

```bash
cd /opt/bach && sudo git pull
sudo bash /opt/bach/deploy/instalar_ec2.sh   # reinstala dependencias y reinicia los servicios
```

`/etc/bach/bach.env` no se sobrescribe.

## Notas

- MLflow 3 rechaza las peticiones cuyo `Host` no está en `--allowed-hosts`. `mlflow_server.sh` agrega
  la IP pública y la privada en cada arranque. Si la Elastic IP se asocia después de instalar, basta con
  `sudo systemctl restart bach-mlflow`.
- La base de datos de MLflow está en `/var/lib/bach/mlflow.db`, en el disco EBS: se conserva al detener
  e iniciar la instancia, pero no si se termina.
