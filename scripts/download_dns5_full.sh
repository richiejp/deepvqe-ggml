#!/usr/bin/env bash
# Download the FULL DNS Challenge 5 training dataset.
#
# Skips anything already downloaded by download_dns5_minimal.sh:
#   - VCTK clean speech (Track1_Headset)
#   - audioset_000, freesound_000 noise shards
#   - impulse_responses_000
#
# New downloads:
#   - 6 AudioSet noise shards (001-006)
#   - 1 Freesound noise shard (001)
#   - 8 clean speech corpora (VocalSet, emotional, read, german, french,
#     spanish, italian, russian)
#
# Total new download: ~600+ GB compressed, ~800+ GB unpacked
#
# Usage:
#   ./scripts/download_dns5_full.sh [output_dir]
#   Default output_dir: ./datasets_fullband

set -euo pipefail

OUT="${1:-./datasets_fullband}"
DNS5="https://dnschallengepublic.blob.core.windows.net/dns5archive/V5_training_dataset"
DNS4="https://dns4public.blob.core.windows.net/dns4archive/datasets_fullband"

mkdir -p "$OUT"/{clean_fullband,noise_fullband,tmp}

echo "=== Downloading DNS Challenge 5 — FULL dataset ==="
echo "Output: $OUT"
echo ""

# ---------------------------------------------------------------------------
# 1. Remaining noise shards (audioset_001-006, freesound_001)
# ---------------------------------------------------------------------------
echo "--- Noise shards (remaining 7 shards) ---"
for shard in audioset_001 audioset_002 audioset_003 audioset_004 audioset_005 audioset_006 freesound_001; do
    url="$DNS4/noise_fullband/datasets_fullband.noise_fullband.${shard}.tar.bz2"
    marker="$OUT/noise_fullband/.${shard}.done"
    if [ -f "$marker" ]; then
        echo "  [skip] $shard (already extracted)"
    else
        echo "  [download+extract] $shard ..."
        curl -L --retry 3 "$url" | tar -C "$OUT" -xjf -
        touch "$marker"
    fi
done
echo "  [done] All noise shards extracted to $OUT/noise_fullband/"

# ---------------------------------------------------------------------------
# 2. Clean speech — Track 1 (Headset) format (.tgz split parts)
# ---------------------------------------------------------------------------

download_headset_speech() {
    local name="$1"
    shift
    local parts=("$@")

    local dest_dir="$OUT/clean_fullband/$name"
    local marker="$OUT/clean_fullband/.${name}.done"

    if [ -f "$marker" ]; then
        echo "  [skip] $name (already extracted)"
        return
    fi

    # Download all parts
    for part in "${parts[@]}"; do
        local url="$DNS5/Track1_Headset/${name}.tgz.${part}"
        local dest="$OUT/tmp/${name}.tgz.${part}"
        if [ -f "$dest" ]; then
            echo "  [skip] $name $part (already downloaded)"
        else
            echo "  [download] $name $part ..."
            curl -L --retry 3 -C - "$url" -o "$dest"
        fi
    done

    # Extract
    if [ ${#parts[@]} -eq 1 ] && [ "${parts[0]}" = "single" ]; then
        # Single .tgz file (no split)
        echo "  [extract] $name ..."
        tar -C "$OUT/clean_fullband" -xzf "$OUT/tmp/${name}.tgz.single"
    else
        echo "  [extract] $name (reassembling ${#parts[@]} parts) ..."
        cat "$OUT"/tmp/${name}.tgz.part* | tar -C "$OUT/clean_fullband" -xzf -
    fi
    touch "$marker"
    echo "  [done] $name"
}

# For single-file .tgz downloads (no split parts)
download_headset_speech_single() {
    local name="$1"

    local marker="$OUT/clean_fullband/.${name}.done"
    if [ -f "$marker" ]; then
        echo "  [skip] $name (already extracted)"
        return
    fi

    local url="$DNS5/Track1_Headset/${name}.tgz"
    local dest="$OUT/tmp/${name}.tgz"
    if [ ! -f "$dest" ]; then
        echo "  [download] $name ..."
        curl -L --retry 3 -C - "$url" -o "$dest"
    fi

    echo "  [extract] $name ..."
    tar -C "$OUT/clean_fullband" -xzf "$dest"
    touch "$marker"
    echo "  [done] $name"
}

# Skip VCTK — already downloaded by minimal script
echo ""
echo "--- Clean speech corpora ---"
echo "  [skip] vctk_wav48_silence_trimmed (from minimal download)"

# VocalSet (~1 GB)
echo ""
echo "--- VocalSet_48kHz_mono (~1 GB) ---"
download_headset_speech_single "VocalSet_48kHz_mono"

# emotional_speech (~2.4 GB)
echo ""
echo "--- emotional_speech (~2.4 GB) ---"
download_headset_speech_single "emotional_speech"

# russian_speech (~12 GB)
echo ""
echo "--- russian_speech (~12 GB) ---"
download_headset_speech_single "russian_speech"

# italian_speech (~42 GB, 4 parts)
echo ""
echo "--- italian_speech (~42 GB) ---"
download_headset_speech "italian_speech" partaa partab partac partad

# french_speech (~62 GB, 6 parts)
echo ""
echo "--- french_speech (~62 GB) ---"
download_headset_speech "french_speech" partaa partab partac partad partae partah

# spanish_speech (~65 GB, 7 parts)
echo ""
echo "--- spanish_speech (~65 GB) ---"
download_headset_speech "spanish_speech" partaa partab partac partad partae partaf partag

# read_speech (~299 GB, 21 parts)
echo ""
echo "--- read_speech (~299 GB) ---"
download_headset_speech "read_speech" partaa partab partac partad partae partaf partag partah partai partaj partak partal partam partan partao partap partaq partar partas partat partau

# german_speech (~319 GB, 22 parts)
echo ""
echo "--- german_speech (~319 GB) ---"
download_headset_speech "german_speech" partaa partab partac partad partae partaf partag partah partai partaj partak partal partam partan partao partap partaq partar partas partat partau partav partaw

# ---------------------------------------------------------------------------
# Cleanup temp files
# ---------------------------------------------------------------------------
echo ""
echo "--- Cleanup ---"
echo "Removing temp download parts ..."
rm -rf "$OUT/tmp"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "=== Download complete ==="
echo ""
echo "Directory structure:"
echo "  $OUT/clean_fullband/"
echo "    vctk_wav48_silence_trimmed/   (~27 GB, from minimal download)"
echo "    VocalSet_48kHz_mono/          (~1 GB)"
echo "    emotional_speech/             (~2.4 GB)"
echo "    russian_speech/               (~12 GB)"
echo "    italian_speech/               (~42 GB)"
echo "    french_speech/                (~62 GB)"
echo "    spanish_speech/               (~65 GB)"
echo "    read_speech/                  (~299 GB)"
echo "    german_speech/                (~319 GB)"
echo "  $OUT/noise_fullband/            (~58 GB, all 9 shards)"
echo "  $OUT/impulse_responses/         (~5.9 GB)"
echo ""
echo "Total: ~890 GB"
echo ""
echo "Note: Audio is 48 kHz WAV. The data pipeline resamples to 16 kHz automatically."
