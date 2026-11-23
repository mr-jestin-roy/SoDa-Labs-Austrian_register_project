#!/bin/bash
# bringup.sh — run on the Mac once `ssh RTB_backed_vlm_monashg2xlarge` works.
#   ./bringup.sh            DNS fix -> data check/upload -> setup (background) on the host
# Then watch:  ssh RTB_backed_vlm_monashg2xlarge 'tail -f /mnt/logs/setup.log'
set -euo pipefail

HOST=RTB_backed_vlm_monashg2xlarge
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new $HOST"

echo "== 1/5 reachability"
$SSH 'hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; lsblk -o NAME,SIZE,MOUNTPOINT | grep -v loop'

echo "== 2/5 /mnt"
if ! $SSH 'mountpoint -q /mnt'; then
  echo "/mnt is not mounted on the new instance. Mount the 2 TB volume (see lsblk above), then rerun."; exit 1
fi
$SSH 'sudo chown -R $USER:$USER /mnt; mkdir -p /mnt/data /mnt/logs /mnt/infra; df -h /mnt | tail -1'

echo "== 3/5 DNS"
scp -q "$HERE/fix_dns.sh" "$HERE/setup_a100.sh" "$HOST:/mnt/infra/"
$SSH 'bash /mnt/infra/fix_dns.sh'

echo "== 4/5 data (upload only what is missing)"
DATA_LOCAL=(
  "$REPO/Reichstelefonbuch_1942/RTB_4_Karbow-Quittainen.pdf"
  "$REPO/Reichstelefonbuch_1942/digitizing python scripts/cv_segment_test.py"
  "$REPO/Reichstelefonbuch_1942/digitizing python scripts/ocr_cells_chandra.py"
  "$REPO/Reichstelefonbuch_1942/digitizing python scripts/orchestrate_pipeline.py"
  "$HERE/data/pernegg_rows.csv"
  "$HERE/data/rtb4_p1266_1285_reference.csv"
)
for f in "${DATA_LOCAL[@]}"; do
  b="$(basename "$f")"
  lsize=$(stat -f%z "$f")
  rsize=$($SSH "stat -c%s '/mnt/data/$b' 2>/dev/null || echo 0")
  if [ "$lsize" = "$rsize" ]; then echo "present  $b"; else echo "upload   $b"; scp -q "$f" "$HOST:/mnt/data/$b"; fi
done

echo "== 5/5 setup (background)"
$SSH 'nohup bash /mnt/infra/setup_a100.sh > /mnt/logs/setup.log 2>&1 & echo "started; tail -f /mnt/logs/setup.log"'
