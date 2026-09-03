#!/usr/bin/env bash
#
# Build and deploy the backend to Cloud Run.
#
# Configuration is loaded automatically from .env in the same directory as
# this script.
#

set -euo pipefail

# ------------------------------------------------------------
# Load .env
# ------------------------------------------------------------

# Resolve the directory containing deploy.sh, so the script can also be run
# from another working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"

if [ ! -f "$ENV_FILE" ]; then
  echo "ERROR: Environment file not found:"
  echo "  ${ENV_FILE}"
  exit 1
fi

echo "Loading environment variables from ${ENV_FILE}..."

# Automatically export every variable loaded from .env.
set -a

# shellcheck disable=SC1090
. "$ENV_FILE"

set +a

# ------------------------------------------------------------
# Defaults
# ------------------------------------------------------------

# These values are used only when they are absent or empty in .env.
REGION="${REGION:-asia-southeast2}"
VERTEX_REGION="${VERTEX_REGION:-global}"
GEMINI_MODEL="${GEMINI_MODEL:-gemini-3.7-flash}"
CORS_ORIGINS="${CORS_ORIGINS:-}"
MIN_INSTANCES="${MIN_INSTANCES:-1}"
MAX_INSTANCES="${MAX_INSTANCES:-10}"

# ------------------------------------------------------------
# Validate required variables
# ------------------------------------------------------------

: "${PROJECT_ID:?PROJECT_ID is missing from .env}"
: "${REPOSITORY:?REPOSITORY is missing from .env}"
: "${IMAGE_NAME:?IMAGE_NAME is missing from .env}"
: "${SERVICE_NAME:?SERVICE_NAME is missing from .env}"
: "${SERVICE_ACCOUNT:?SERVICE_ACCOUNT is missing from .env}"
: "${INSTANCE_CONNECTION_NAME:?INSTANCE_CONNECTION_NAME is missing from .env}"
: "${DB_NAME:?DB_NAME is missing from .env}"
: "${DB_USER:?DB_USER is missing from .env}"
: "${DB_PASS:?DB_PASS is missing from .env}"

IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:latest"
SERVICE_ACCOUNT_PATH="projects/${PROJECT_ID}/serviceAccounts/${SERVICE_ACCOUNT}"

echo
echo "Deployment configuration:"
echo "  Project:              ${PROJECT_ID}"
echo "  Region:               ${REGION}"
echo "  Repository:           ${REPOSITORY}"
echo "  Image:                ${IMAGE}"
echo "  Cloud Run service:    ${SERVICE_NAME}"
echo "  Service account:      ${SERVICE_ACCOUNT}"
echo "  Cloud SQL connection: ${INSTANCE_CONNECTION_NAME}"
echo "  Database:             ${DB_NAME}"
echo "  Database user:        ${DB_USER}"
echo "  Vertex region:        ${VERTEX_REGION}"
echo "  Gemini model:         ${GEMINI_MODEL}"
echo "  Min instances:        ${MIN_INSTANCES}"
echo "  Max instances:        ${MAX_INSTANCES}"
echo

# Deliberately do not print DB_PASS.

gcloud config set project "$PROJECT_ID"

# ------------------------------------------------------------
# One-time setup
# ------------------------------------------------------------
#
# Uncomment and run once if these APIs and IAM roles have not been configured.
#
# gcloud services enable \
#   run.googleapis.com \
#   cloudbuild.googleapis.com \
#   artifactregistry.googleapis.com \
#   sqladmin.googleapis.com \
#   aiplatform.googleapis.com
#
# Runtime roles:
#
# gcloud projects add-iam-policy-binding "$PROJECT_ID" \
#   --member "serviceAccount:${SERVICE_ACCOUNT}" \
#   --role roles/cloudsql.client
#
# gcloud projects add-iam-policy-binding "$PROJECT_ID" \
#   --member "serviceAccount:${SERVICE_ACCOUNT}" \
#   --role roles/aiplatform.user
#
# Build roles:
#
# gcloud projects add-iam-policy-binding "$PROJECT_ID" \
#   --member "serviceAccount:${SERVICE_ACCOUNT}" \
#   --role roles/cloudbuild.builds.builder
#
# gcloud projects add-iam-policy-binding "$PROJECT_ID" \
#   --member "serviceAccount:${SERVICE_ACCOUNT}" \
#   --role roles/artifactregistry.writer

# ------------------------------------------------------------
# Build
# ------------------------------------------------------------

echo "Building ${IMAGE}..."

gcloud builds submit \
  --region "$REGION" \
  --tag "$IMAGE" \
  --service-account "$SERVICE_ACCOUNT_PATH" \
  --default-buckets-behavior "REGIONAL_USER_OWNED_BUCKET"

# ------------------------------------------------------------
# Deploy
# ------------------------------------------------------------

# The ^@^ prefix tells gcloud to use @ instead of comma as the environment
# variable delimiter.
ENV_VARS="^@^GCP_PROJECT_ID=${PROJECT_ID}"
ENV_VARS="${ENV_VARS}@VERTEX_REGION=${VERTEX_REGION}"
ENV_VARS="${ENV_VARS}@GEMINI_MODEL=${GEMINI_MODEL}"
ENV_VARS="${ENV_VARS}@INSTANCE_CONNECTION_NAME=${INSTANCE_CONNECTION_NAME}"
ENV_VARS="${ENV_VARS}@DB_NAME=${DB_NAME}"
ENV_VARS="${ENV_VARS}@DB_USER=${DB_USER}"
ENV_VARS="${ENV_VARS}@DB_PASS=${DB_PASS}"
ENV_VARS="${ENV_VARS}@CORS_ORIGINS=${CORS_ORIGINS}"

echo "Deploying ${SERVICE_NAME}..."

gcloud run deploy "$SERVICE_NAME" \
  --image "$IMAGE" \
  --platform managed \
  --region "$REGION" \
  --set-env-vars "$ENV_VARS" \
  --add-cloudsql-instances "$INSTANCE_CONNECTION_NAME" \
  --service-account "$SERVICE_ACCOUNT" \
  --port 8080 \
  --allow-unauthenticated \
  --memory 512Mi \
  --cpu 1 \
  --cpu-boost \
  --timeout 300 \
  --min-instances "$MIN_INSTANCES" \
  --max-instances "$MAX_INSTANCES"

SERVICE_URL="$(
  gcloud run services describe "$SERVICE_NAME" \
    --region "$REGION" \
    --format="value(status.url)"
)"

echo
echo "Service deployed to: ${SERVICE_URL}"
echo
echo "Verify that schema_loaded is true and row_count is 400:"
echo "  curl ${SERVICE_URL}/api/health"