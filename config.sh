# Source this file before launching the application (or Qt Creator).
qcap_project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export QCAP_LOG_LEVEL=0

ulimit -Sn 8192 || { printf '%s\n' 'Cannot set FD soft limit to 8192' >&2; return 1; }
qcap_fd_library="$qcap_project_dir/select2poll/libqcap_lowfd.so"
if [ ! -r "$qcap_fd_library" ]; then
    printf 'Missing FD compatibility library: %s\n' "$qcap_fd_library" >&2
    return 1
fi
# Replace the obsolete select2poll preload; do not suppress FD_SET bounds checks.
export LD_PRELOAD="$qcap_fd_library"
export LD_LIBRARY_PATH="$qcap_project_dir/lib:$qcap_project_dir/qdeep/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset qcap_fd_library qcap_project_dir
