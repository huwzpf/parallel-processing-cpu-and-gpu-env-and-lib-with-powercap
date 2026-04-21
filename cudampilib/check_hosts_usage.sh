#!/bin/bash

set -u

HOSTFILE="${1:-$(dirname "$0")/hostfile}"
SSH_TIMEOUT="${SSH_TIMEOUT:-8}"
TOP_ROWS="${TOP_ROWS:-15}"
CURRENT_USER="$(id -un)"

if [ ! -f "$HOSTFILE" ]; then
    echo "Hostfile not found: $HOSTFILE" >&2
    exit 1
fi

hosts=()
while read -r host _; do
    case "$host" in
        ""|\#*) continue ;;
    esac
    hosts+=("$host")
done < "$HOSTFILE"

if [ "${#hosts[@]}" -eq 0 ]; then
    echo "No hosts found in $HOSTFILE" >&2
    exit 1
fi

for host in "${hosts[@]}"; do
    echo "================================================================"
    echo "HOST: $host"
    echo "================================================================"

    ssh \
        -o BatchMode=yes \
        -o ConnectTimeout="$SSH_TIMEOUT" \
        -o StrictHostKeyChecking=no \
        "$host" \
        "CURRENT_USER='$CURRENT_USER' TOP_ROWS='$TOP_ROWS' bash -s" <<'REMOTE'
set -u

echo "[connected users]"
if command -v who >/dev/null 2>&1; then
    who || true
else
    echo "who command not available"
fi

echo
echo "[users other than ${CURRENT_USER}]"
if command -v who >/dev/null 2>&1; then
    others="$(who | awk -v me="$CURRENT_USER" '$1 != me { print }')"
    if [ -n "$others" ]; then
        echo "$others"
    else
        echo "none detected via who"
    fi
else
    echo "cannot determine"
fi

echo
echo "[nvidia-smi]"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
else
    echo "nvidia-smi not available"
fi

echo
echo "[gpu processes]"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true
else
    echo "nvidia-smi not available"
fi

echo
echo "[top cpu processes]"
if command -v top >/dev/null 2>&1; then
    top -b -n 1 -o %CPU | head -n "$TOP_ROWS" || true
else
    echo "top not available"
fi
REMOTE

    status=$?
    if [ "$status" -ne 0 ]; then
        echo
        echo "Failed to inspect $host over SSH (exit status $status)."
    fi

    echo
done
