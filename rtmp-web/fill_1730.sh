#!/bin/bash
# one-off (3 Oct): after the first hd-retime pass, replace 17:30-18:30 from the SD card
cd /home/ubuntu/rtmp-web
while [ "$(systemctl show -p ActiveState --value hd-retime.service)" != inactive ]; do sleep 30; done
echo "$(date +%H:%M:%S) retiming done; starting the 17:30-18:30 fill"
for i in $(seq 1 10); do
  out=$(node sd_fill.js state/hd-device.json cam1 --now --from=17:30 --to=18:30 --day=2026-10-03 2>&1); echo "$out"
  echo "$out" | grep -q "nothing to fill" && { echo "$(date +%H:%M:%S) FINISHED"; exit 0; }
  echo "$out" | grep -q "stalled lately" && sleep 300
  sleep 20
done
echo "$(date +%H:%M:%S) gave up after 10 runs"
