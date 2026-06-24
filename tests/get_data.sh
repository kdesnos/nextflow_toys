#!/bin/bash

cd "$(dirname "$0")/.." || exit

base_url="https://kdesnos.fr/wp-content/uploads/nextflow_traces/"
base_dir="./dat"

files=(
  "250510_210912_CELEBI/karol_210912_ult_2025-05-10_10_30_18_report.html"
  "250511_250201_CELEBI/karol_250201_2025-05-11_09_56_28_report.html"
  "250508_250313_CELEBI/karol_250313_2025-05-08_10_36_28_report.html"
  "250514_250106_CELEBI/karol_250106_ult_2025-05-14_09_41_50_report.html"
  "250515_241226_CELEBI/karol_241226_ult_2025-05-15_13_41_42_report.html"
  "250522_241027_CELEBI/karol_241027_ult_2025-05-22_10_05_56_report.html"
  "250523_241014_CELEBI/karol_241014_ult_2025-05-23_12_12_20_report.html"
  "250616_241226_CELEBI_mold/karol_241226_ult_2025-06-16_10_22_54_report.html"
  "250616_210912_CELEBI_mold/karol_210912_ult_2025-06-16_13_46_53_report.html"
  "250510_210912_CELEBI/karol_210912_ult_2025-05-10_10_30_18_log.log"
  "250511_250201_CELEBI/karol_250201_2025-05-11_09_56_28_log.log"
  "250508_250313_CELEBI/karol_250313_2025-05-08_10_36_28_log.log"
  "250514_250106_CELEBI/karol_250106_ult_2025-05-14_09_41_50_log.log"
  "250515_241226_CELEBI/karol_241226_ult_2025-05-15_13_41_42_log.log"
  "250522_241027_CELEBI/karol_241027_ult_2025-05-22_10_05_56_log.log"
  "250523_241014_CELEBI/karol_241014_ult_2025-05-23_12_12_20_log.log"
  "250616_241226_CELEBI_mold/karol_241226_ult_2025-06-16_10_22_54_log.log"
  "250616_210912_CELEBI_mold/karol_210912_ult_2025-06-16_13_46_53_log.log"
)

for file in "${files[@]}"; do
  local_path="${base_dir}/${file}"
  mkdir -p "$(dirname "$local_path")"
  curl -o "$local_path" "${base_url}${file}"
  echo "Téléchargé : ${local_path}"
done