# Multi-stage Docker build for Shevelev proof
# Stage 1: Build and cache Mathlib (slow, but cached in layers)
# Stage 2: Add proof code and verify (small, fast, immutable)

FROM leanprover/lean4:v4.30.0 AS builder

WORKDIR /build

# Copy Lake configuration and dependencies
COPY lakefile.toml lean-toolchain lake-manifest.json ./

# Update Lake and build Mathlib
# This layer is cached, so rebuilds are fast if these files don't change
RUN lake update && lake build Mathlib

# Final stage: minimal image with cached Mathlib + proof
FROM leanprover/lean4:v4.30.0

WORKDIR /proof

# Copy cached .lake from builder (contains compiled Mathlib)
COPY --from=builder /build/.lake .lake

# Copy proof code and documentation
COPY Shevelev.lean Proof.lean README.md CONTRIBUTING.md ./
COPY get_cache.py ./

# Verify proof compiles (fail-fast in build)
RUN lake env lean Shevelev.lean

# Default: interactive bash shell in proof directory
CMD ["bash"]
