#!/usr/bin/env bash

# Prefer driver libraries supplied by the host/container without assuming a
# particular Conda installation. This function is intentionally safe to call
# on CPU-only hosts used for launcher validation.
prefer_kernel_nvidia_driver() {
  local candidate
  for candidate in /usr/local/nvidia/lib64 /usr/local/nvidia/lib; do
    if [[ -d "${candidate}" ]]; then
      LD_LIBRARY_PATH="${candidate}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi
  done
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
  export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
}
