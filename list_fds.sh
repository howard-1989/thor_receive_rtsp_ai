#!/bin/bash
# list_fds.sh — Dump FD info for a running QtQcapMultiClientDemo process
set -e
for prog in QtQcapMultiClientDemo_17kps QtQcapMultiClientDemo_people QtQcapMultiClientDemo_plate; do
    pid=$(pidof "$prog" 2>/dev/null)
    if [ -n "$pid" ]; then
        echo "=========================================="
        echo "  $prog (PID=$pid)"
        echo "=========================================="
        echo "Open FD count: $(ls -1 /proc/$pid/fd/ 2>/dev/null | wc -l)"
        echo "Highest FD:    $(ls /proc/$pid/fd/ 2>/dev/null | sort -n | tail -5 | tr '\n' ' ')"
        echo ""
        echo "--- FD type distribution (top 30) ---"
        ls -la /proc/$pid/fd/ 2>/dev/null | awk '{print $NF}' | sort | uniq -c | sort -rn | head -30
        echo ""
        echo "--- Highest 10 FDs ---"
        for fd in $(ls /proc/$pid/fd/ 2>/dev/null | sort -n | tail -10); do
            echo "  fd=$fd -> $(readlink /proc/$pid/fd/$fd 2>/dev/null)"
        done
        exit 0
    fi
done
echo "No target QtQcapMultiClientDemo process found."
echo "Usage: start one of the demos, then run this script."
