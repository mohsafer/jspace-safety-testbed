#!/bin/bash
# Watchdog v3 for the new-server campaign (pidfile-based liveness, same
# contract as scripts/watchdog.sh — never pgrep by script name; the launcher
# wrappers embed script strings and poison string matching).
#
# Usage: scripts/watchdog6.sh <programN> [max_restarts]
#   watches artifacts/pid_gpu_program<N>, restarts scripts/gpu_program<N>.sh
#   until artifacts/gpu_program<N>.log contains GPU_PROGRAM<N>_COMPLETE.
#
# GPU workload monitoring (PI requirement): every cycle records
# utilization/memory to artifacts/gpu_workload.csv and raises advisory
# conditions in artifacts/watchdog.log:
#   STALL(n)  program alive but GPU idle (util 0%, <100 MiB) for n cycles —
#             expected during CPU-eval phases of a chain, so ADVISORY ONLY
#             (no auto-kill: killing a live CPU eval would corrupt the chain)
#   ORPHAN    GPU memory held while no program pidfile is alive — the
#             classic pre-OOM state; orphans are cleaned before any restart
#   OOMGROW   memory > 23 GiB (24 GB card) — fragmentation/OOM precursor
# Restart policy unchanged: max restarts, orphan pkill before each relaunch.
cd /users/mosafer/spacepriv8
PROG="${1:?usage: watchdog6.sh <programN> [max_restarts]}"
MAX="${2:-3}"
POLL="${POLL_SEC:-600}"
WD=artifacts/watchdog.log
CSV=artifacts/gpu_workload.csv
tries=0
stall=0
log() { echo "[watchdog$PROG $(date '+%m-%d %H:%M:%S')] $*" >> "$WD"; }
alive() {
  [ -f "artifacts/pid_gpu_program$PROG" ] || return 1
  kill -0 "$(cat artifacts/pid_gpu_program$PROG)" 2>/dev/null
}
[ -f "$CSV" ] || echo "timestamp,program,alive,util_pct,mem_mib" >> "$CSV"
log "=== watchdog$PROG v3 start (pid $$, max $MAX restarts, poll ${POLL}s, GPU monitor on) ==="
while true; do
  if grep -q "GPU_PROGRAM${PROG}_COMPLETE" "artifacts/gpu_program$PROG.log" 2>/dev/null; then
    log "SUCCESS: GPU_PROGRAM${PROG}_COMPLETE marker found"
    break
  fi
  A=$(alive && echo yes || echo no)
  U=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null)
  UTIL=$(echo "$U" | cut -d, -f1 | tr -d ' ')
  MEM=$(echo "$U" | cut -d, -f2 | tr -d ' ')
  echo "$(date '+%Y-%m-%d %H:%M:%S'),$PROG,$A,${UTIL:-NA},${MEM:-NA}" >> "$CSV"
  # advisory GPU-workload conditions
  if [ "$A" = "yes" ] && [ "${UTIL:-0}" -le 0 ] && [ "${MEM:-0}" -lt 100 ]; then
    stall=$((stall+1))
    [ $((stall % 3)) -eq 1 ] && log "ADVISORY STALL($stall): program$PROG alive but GPU idle ${stall}x${POLL}s (expected during CPU-eval stages; escalate if it outlasts the longest CPU eval ~75min)"
  else
    stall=0
  fi
  if [ "$A" = "no" ] && [ "${MEM:-0}" -gt 1000 ]; then
    log "ADVISORY ORPHAN: ${MEM} MiB GPU memory held with no program alive (cleaned before any restart)"
  fi
  if [ "${MEM:-0}" -gt 23500 ]; then
    log "ADVISORY OOMGROW: GPU memory ${MEM} MiB near the 24576 MiB limit"
  fi

  if ! alive; then
    tries=$((tries+1))
    if [ "$tries" -gt "$MAX" ]; then
      log "GIVE UP: program$PROG failed $MAX attempts — manual attention needed"
      break
    fi
    log "program$PROG not running — (re)starting (attempt $tries/$MAX)"
    # orphan cleanup first: killed parents leave pythons holding VRAM.
    # Kill ONLY processes actually holding GPU memory (by pid from the
    # driver) — the old name-based `pkill -f "src\.run_|..."` also SIGKILLed
    # innocent CPU jobs (e.g. a running src.run_p1 recompute), which is the
    # exact by-script-name anti-pattern AGENTS.md forbids.
    for opid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
      kill -9 "$opid" 2>/dev/null
    done
    rm -f "artifacts/pid_gpu_program$PROG"
    sleep 5
    setsid nohup bash "scripts/gpu_program$PROG.sh" </dev/null >/dev/null 2>&1 &
    sleep 120
  fi
  sleep "$POLL"
done
