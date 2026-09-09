#!/usr/bin/env bash
# ============================================================
# AKASHA Frontend — AWS Amplify Deploy Helper
# ============================================================
# This script is a convenience wrapper for local builds only.
# Production deploys happen automatically via the Amplify Console
# CI/CD pipeline on every push to the main branch.
#
# Usage (manual / CI):
#   ./scripts/deploy-frontend.sh [environment]
#
# environment: "production" (default) | "staging"
# ============================================================

set -euo pipefail

ENVIRONMENT="${1:-production}"
FRONTEND_DIR="$(cd "$(dirname "$0")/../frontend" && pwd)"

echo "▶  AKASHA Frontend Build — ${ENVIRONMENT}"
echo "   Directory: ${FRONTEND_DIR}"

# ────────────────────────────────────────────────────────────
# Install dependencies
# ────────────────────────────────────────────────────────────
echo ""
echo "▶  Installing npm dependencies…"
cd "${FRONTEND_DIR}"
npm ci

# ────────────────────────────────────────────────────────────
# Build
# ────────────────────────────────────────────────────────────
echo ""
echo "▶  Building for ${ENVIRONMENT}…"

if [ "${ENVIRONMENT}" = "production" ]; then
  # Vite automatically picks up .env.production when NODE_ENV=production
  NODE_ENV=production npm run build
else
  npm run build
fi

echo ""
echo "✔  Build complete — output in ${FRONTEND_DIR}/dist"
echo ""
echo "──────────────────────────────────────────────────────"
echo "  Deployment options:"
echo ""
echo "  AWS Amplify Console (recommended):"
echo "    → Push to the 'main' branch. Amplify CI/CD will"
echo "      automatically build and deploy."
echo ""
echo "  Amplify CLI (manual):"
echo "    amplify publish"
echo ""
echo "  Firebase Hosting (legacy fallback):"
echo "    firebase deploy --only hosting"
echo "──────────────────────────────────────────────────────"
