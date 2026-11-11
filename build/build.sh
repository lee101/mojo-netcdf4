#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$repo_dir/dist"
# MNC_SRC/MNC_OUT let a test build a variant of the kernel source (for example
# with the parallel threshold disabled) without touching the shipped library.
src="${MNC_SRC:-$repo_dir/src/netcdf4.mojo}"
out="${MNC_OUT:-$repo_dir/dist/libmojo-netcdf4.so}"
mojo build --emit shared-lib "$src" -o "$out"
