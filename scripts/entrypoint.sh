#!/usr/bin/env bash
# Docker entrypoint: mount .sqsh datasets if present, then exec the command.
set -euo pipefail

SQSH="/workspace/deepvqe/datasets_fullband/sqsh"
MNT="/data"

# Mount squashfs images to /data/<name> if /dev/fuse is available
if [ -c /dev/fuse ] && [ -d "$SQSH" ]; then
    for sqsh_file in "$SQSH"/*.sqsh; do
        [ -f "$sqsh_file" ] || continue
        name="$(basename "$sqsh_file" .sqsh)"
        target="$MNT/$name"
        # Skip broken sqsh files (< 10KB)
        size="$(stat -c%s "$sqsh_file")"
        if [ "$size" -lt 10000 ]; then
            echo "[entrypoint] skipping $name.sqsh ($size bytes, likely broken)"
            continue
        fi
        mkdir -p "$target"
        echo "[entrypoint] mounting $name.sqsh -> $target ($size bytes)"
        squashfuse "$sqsh_file" "$target"
    done
else
    echo "[entrypoint] /dev/fuse or sqsh dir not available, skipping sqsh mounts"
fi

exec "$@"
