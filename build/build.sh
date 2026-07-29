#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$repo_dir/dist"
mojo build --emit shared-lib "$repo_dir/src/netcdf4.mojo" \
  -o "$repo_dir/dist/libmojo-netcdf4.so"
