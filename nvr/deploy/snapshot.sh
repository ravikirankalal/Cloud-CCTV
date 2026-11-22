#!/bin/bash
# Periodic consistent snapshot of nvr.db for the REMOTE detector (micro #2) to read over the NFS export.
# VACUUM INTO makes a clean, WAL-free copy of the live DB; the atomic rename means an NFS reader never
# sees a half-written file. Kept light (Nice + idle IO in the unit) so it never competes with recording.
# The remote detector opens this snapshot immutable and reopens it each poll, so a ~15s metadata lag is
# all it costs — fine for a post-hoc consumer.
set -u
DB=/home/ubuntu/nvr-data/nvr.db
SNAP=/home/ubuntu/nvr-data/nvr-snap.db
INT=${SNAP_INTERVAL:-15}
while true; do
  rm -f "$SNAP.tmp"
  if sqlite3 "$DB" "VACUUM INTO '$SNAP.tmp'" 2>/dev/null; then
    mv -f "$SNAP.tmp" "$SNAP"
  fi
  sleep "$INT"
done
