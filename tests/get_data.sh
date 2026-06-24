# 1. On s'assure d'être dans le bon dossier et on prépare les variables
base_url="https://kdesnos.fr/wp-content/uploads/nextflow_traces/"
base_dir="./dat"

files=(
  "250510_210912_CELEBI/karol_210912_ult_2025-05-10_10_30_18_report.html"
  "250511_250201_CELEBI/karol_250201_2025-05-11_09_56_28_report.html"
  "250508_250313_CELEBI/karol_250313_2025-05-08_10_36_28_report.html"
  "250514_250106_CELEBI/karol_250106_ult_2025-05-14_09_41_50_report.html"
  "250510_210912_CELEBI/karol_210912_ult_2025-05-10_10_30_18_log.log"
  "250511_250201_CELEBI/karol_250201_2025-05-11_09_56_28_log.log"
  "250508_250313_CELEBI/karol_250313_2025-05-08_10_36_28_log.log"
  "250514_250106_CELEBI/karol_250106_ult_2025-05-14_09_41_50_log.log"
)

# 2. On boucle pour tout télécharger d'un coup
for file in "${files[@]}"; do
  local_path="${base_dir}/${file}"
  mkdir -p "$(dirname "$local_path")"
  curl -o "$local_path" "${base_url}${file}"
  echo "Téléchargé : ${local_path}"
done