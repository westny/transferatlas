#!/usr/bin/env bash

set -euo pipefail

datasets=(
  A43
  SIND
  ad4che
  apollo
  argoverse
  av2
  eth
  exiD
  highD
  hotel
  i80
  inD
  interact
  lyft
  nuscenes
  openDD
  rounD
  uniD
  univ
  us101
  vod
  waymo
  zara1
  zara2
)

for dataset in "${datasets[@]}"; do
  printf '\nComputing latents for %s\n' "$dataset"
  uv run transferatlas-latents \
    --config trace \
    --checkpoint artifacts/checkpoints/trace32.pt \
    --dataset "$dataset" \
    --use-cuda true \
    --dry-run false
done
