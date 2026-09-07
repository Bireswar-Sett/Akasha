#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRONTEND_DIR="$ROOT_DIR/frontend"
BACKEND_DIR="$ROOT_DIR/backend"

PROJECT_ID="${FIREBASE_PROJECT_ID:-akasha-v1}"
REGION="${CLOUD_RUN_REGION:-us-central1}"
SERVICE_NAME="${CLOUD_RUN_SERVICE:-akasha-backend}"
HOSTING_SITE="${FIREBASE_HOSTING_SITE:-akasha}"

echo "========================================"
echo "        AKASHA AUTOMATED DEPLOY"
echo "========================================"
echo
echo "Project : $PROJECT_ID"
echo "Region  : $REGION"
echo "Service : $SERVICE_NAME"
echo

# --------------------------------------------------
# 1. Check required commands
# --------------------------------------------------

for command_name in gcloud firebase npm; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "ERROR: Required command not found: $command_name" >&2
        exit 1
    fi
done

# --------------------------------------------------
# 2. Verify authentication
# --------------------------------------------------

echo "[1/5] Checking Google Cloud authentication..."

if ! gcloud auth print-access-token >/dev/null 2>&1; then
    echo "ERROR: Google Cloud authentication failed."
    echo "Run: gcloud auth login"
    exit 1
fi

echo "Google Cloud authentication OK."

echo
echo "[2/5] Checking Firebase authentication..."

if ! firebase projects:list >/dev/null 2>&1; then
    echo "ERROR: Firebase authentication failed."
    echo "Run: firebase login"
    exit 1
fi

echo "Firebase authentication OK."

# --------------------------------------------------
# 3. Select project
# --------------------------------------------------

echo
echo "[3/5] Selecting Google Cloud project..."

gcloud config set project "$PROJECT_ID" >/dev/null

echo "Project selected: $PROJECT_ID"

# --------------------------------------------------
# 4. Deploy backend
# --------------------------------------------------

echo
echo "[4/5] Deploying backend to Cloud Run..."
echo
echo "IMPORTANT:"
echo "Existing Cloud Run environment variables and secrets"
echo "will NOT be modified by this deployment."
echo

gcloud run deploy "$SERVICE_NAME" \
    --source "$BACKEND_DIR" \
    --project "$PROJECT_ID" \
    --region "$REGION" \
    --platform managed \
    --allow-unauthenticated

BACKEND_URL="$(
    gcloud run services describe "$SERVICE_NAME" \
        --project "$PROJECT_ID" \
        --region "$REGION" \
        --format='value(status.url)'
)"

echo
echo "Backend deployed:"
echo "$BACKEND_URL"

# --------------------------------------------------
# 5. Build + deploy frontend
# --------------------------------------------------

echo
echo "[5/5] Building frontend..."

cd "$FRONTEND_DIR"

npm ci
npm run build

echo
echo "Deploying Firebase Hosting..."

firebase use "$PROJECT_ID" --add >/dev/null 2>&1 || true

firebase deploy \
    --project "$PROJECT_ID" \
    --only "hosting:$HOSTING_SITE,storage"

FRONTEND_URL="https://${HOSTING_SITE}.web.app"

# --------------------------------------------------
# Done
# --------------------------------------------------

echo
echo "========================================"
echo "       AKASHA DEPLOYMENT COMPLETE"
echo "========================================"
echo
echo "Frontend:"
echo "$FRONTEND_URL"
echo
echo "Backend:"
echo "$BACKEND_URL"
echo
echo "Cloud Run service:"
echo "$SERVICE_NAME"
echo
echo "Project:"
echo "$PROJECT_ID"
echo