#!/usr/bin/env bash
# Pack DNS5 dataset directories into SquashFS images and mount them.
#
# Creates one .sqsh per subdirectory (clean, noise, rir) for independent
# management. Mounts them back to the original paths via squashfuse so
# training sees no difference.
#
# Usage:
#   ./scripts/pack_squashfs.sh [dataset_dir]
#   Default dataset_dir: ./datasets_fullband
#
# Requirements: mksquashfs, squashfuse (provided by flake.nix)

set -euo pipefail

SRC="${1:-./datasets_fullband}"
SQSH_DIR="${SRC}/sqsh"

mkdir -p "$SQSH_DIR"

pack_and_mount() {
    local name="$1"
    local src_dir="$2"
    local sqsh="$SQSH_DIR/${name}.sqsh"

    if [ ! -d "$src_dir" ]; then
        echo "  [skip] $name — $src_dir does not exist"
        return
    fi

    if [ -f "$sqsh" ]; then
        echo "  [skip] $name — $sqsh already exists"
    else
        local src_size
        src_size=$(du -sh "$src_dir" 2>/dev/null | cut -f1)
        echo "  [pack] $name ($src_size) → $sqsh ..."
        mksquashfs "$src_dir" "$sqsh" -comp zstd -Xcompression-level 3 -no-progress
        local sqsh_size
        sqsh_size=$(du -sh "$sqsh" | cut -f1)
        echo "  [done] $name: $src_size → $sqsh_size"
    fi

    # Check if already mounted
    if mountpoint -q "$src_dir" 2>/dev/null; then
        echo "  [skip] $name — already mounted"
        return
    fi

    # Remove original data and mount squashfs in its place
    echo "  [mount] $name → $src_dir"
    rm -rf "$src_dir"
    mkdir -p "$src_dir"
    squashfuse "$sqsh" "$src_dir"
}

echo "=== Packing DNS5 data into SquashFS ==="
echo "Source: $SRC"
echo "Images: $SQSH_DIR/"
echo ""

pack_and_mount "clean_fullband" "$SRC/clean_fullband"
pack_and_mount "noise_fullband" "$SRC/noise_fullband"
pack_and_mount "impulse_responses" "$SRC/impulse_responses"

echo ""
echo "=== Done ==="
echo ""
echo "Mounted squashfs images:"
mount | grep squashfuse | sed 's/^/  /' || echo "  (none — check above for errors)"
echo ""
echo "To unmount later:"
echo "  fusermount -u $SRC/clean_fullband"
echo "  fusermount -u $SRC/noise_fullband"
echo "  fusermount -u $SRC/impulse_responses"
echo ""
echo "To remount after reboot:"
echo "  squashfuse $SQSH_DIR/clean_fullband.sqsh $SRC/clean_fullband"
echo "  squashfuse $SQSH_DIR/noise_fullband.sqsh $SRC/noise_fullband"
echo "  squashfuse $SQSH_DIR/impulse_responses.sqsh $SRC/impulse_responses"
