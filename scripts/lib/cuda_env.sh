#!/usr/bin/env bash

# Prefer the userspace NVIDIA driver library that matches the loaded kernel
# module. Some AIStudio images register /usr/local/cuda/compat ahead of the
# host-mounted driver, which makes cuInit fail with CUDA error 803.
prefer_kernel_nvidia_driver() {
  local version_line driver_version candidate_dir candidate_file

  if [[ -n "${HOST_NVIDIA_DRIVER_LIB_DIR:-}" ]]; then
    export LD_LIBRARY_PATH="${HOST_NVIDIA_DRIVER_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    echo "nvidia_driver_lib=${HOST_NVIDIA_DRIVER_LIB_DIR} (explicit)"
    return 0
  fi

  [[ -r /proc/driver/nvidia/version ]] || return 0
  version_line=$(head -n 1 /proc/driver/nvidia/version)
  if [[ ! "${version_line}" =~ ([0-9]{3,}\.[0-9]+\.[0-9]+) ]]; then
    return 0
  fi
  driver_version="${BASH_REMATCH[1]}"

  for candidate_dir in \
    /lib64 \
    /usr/lib64 \
    /lib/x86_64-linux-gnu \
    /usr/lib/x86_64-linux-gnu \
    /lib \
    /usr/lib; do
    candidate_file="${candidate_dir}/libcuda.so.${driver_version}"
    if [[ -f "${candidate_file}" ]]; then
      export LD_LIBRARY_PATH="${candidate_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
      echo "nvidia_driver_lib=${candidate_dir} kernel_driver=${driver_version}"
      return 0
    fi
  done

  echo "WARN: no libcuda.so matching kernel driver ${driver_version} was found." >&2
  return 0
}
