FROM nvcr.io/nvidia/pytorch:26.02-py3

# libsndfile for the soundfile Python package, squashfuse for .sqsh dataset mounts
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 squashfuse fuse \
    && rm -rf /var/lib/apt/lists/*

# Install only packages not already in the NGC container
# (numpy, scipy, tensorboard, pyyaml, matplotlib, tqdm are pre-installed)
RUN pip install --no-cache-dir einops pesq pystoi soundfile gguf

WORKDIR /workspace/deepvqe

# Source code is mounted at runtime; only the entrypoint is baked in
COPY scripts/entrypoint.sh /usr/local/bin/entrypoint.sh
ENTRYPOINT ["entrypoint.sh"]
