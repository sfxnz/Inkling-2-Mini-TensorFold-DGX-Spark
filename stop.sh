#!/usr/bin/env bash
# Gracefully stops and removes only the container carrying LABEL and CONTAINER_NAME from recipe.conf.
set -euo pipefail
DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=recipe.conf
source "$DIR/recipe.conf"

if ! grep -Fxq "$CONTAINER_NAME" <<<"$(docker ps -a --filter "label=$LABEL" --format '{{.Names}}')"; then
  echo "no $CONTAINER_NAME on $(hostname -s)"
  exit 0
fi
docker stop -t 60 "$CONTAINER_NAME" >/dev/null
docker rm "$CONTAINER_NAME" >/dev/null
echo "stopped and removed $CONTAINER_NAME on $(hostname -s)"
