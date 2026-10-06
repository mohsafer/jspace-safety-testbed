#!/bin/bash
# Watchdog: keeps the GPU experiment chain (program2 -> program3) running until
# GPU_PROGRAM3_COMPLETE appears. Liveness is checked via PID FILES
# (artifacts/pid_gpu_program2/3) — pgrep-by-name is unreliable here because
# launcher wrapper command lines embed the script names (that bug silently
# disabled the first watchdog for 6h). All stages are resumable; restarts are
# therefore safe. Heartbeats to artifacts/watchdog.log; gives up after MAX
# attempts per program.
cd /users/mosafer/spacepriv8
WD=artifacts/watchdog.log
MAX=5
P2=0; P3=0
log() { echo "[watchdog $(date '+%m-%d %H:%M:%S')] $*" >> "$WD"; }
alive() {  # alive <pidfile> — true iff pidfile exists and its pid is running
  local f="artifacts/pid_$1"
  [ -f "$f" ] || return 1
  kill -0 "$(cat "$f")" 2>/dev/null
}
log "=== watchdog v2 start (pid $$, pidfile-based liveness) ==="
while true; do
  if grep -q GPU_PROGRAM3_COMPLETE artifacts/gpu_program3.log 2>/dev/null; then
    log "SUCCESS: GPU_PROGRAM3_COMPLETE marker found — experiment chain finished"
    break
  fi
  if grep -q GPU_PROGRAM2_COMPLETE artifacts/gpu_program2.log 2>/dev/null; then
    if ! alive gpu_program3; then
      P3=$((P3+1))
      if [ "$P3" -gt "$MAX" ]; then
        log "GIVE UP: program3 failed $MAX attempts — manual attention needed"
        break
      fi
      log "program2 complete; launching program3 (attempt $P3/$MAX)"
      setsid nohup bash scripts/gpu_program3.sh </dev/null >/dev/null 2>&1 &
      sleep 120
    fi
  else
    if ! alive gpu_program2; then
      P2=$((P2+1))
      if [ "$P2" -gt "$MAX" ]; then
        log "GIVE UP: program2 failed $MAX attempts — manual attention needed"
        break
      fi
      log "program2 not running (pidfile check) — (re)starting (attempt $P2/$MAX)"
      # clear orphaned workers first (killed parents leave pythons holding VRAM;
      # a fresh instance then OOMs and the whole chain spirals)
      pkill -9 -f "src\.run_|src\.eval_|src\.p3_posthoc|src\.make_" 2>/dev/null
      rm -f artifacts/pid_gpu_program2
      sleep 5
      setsid nohup bash scripts/gpu_program2.sh </dev/null >/dev/null 2>&1 &
      sleep 120
    fi
  fi
  U=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | tr '\n' ' ')
  P2A=$(alive gpu_program2 && echo yes || echo no)
  P3A=$(alive gpu_program3 && echo yes || echo no)
  log "heartbeat gpu=[$U] program2_alive=$P2A program3_alive=$P3A p2_tries=$P2 p3_tries=$P3"
  sleep 600
done
