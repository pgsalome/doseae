#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
opentps_root="${repo_root}/external/OpenTPS"
patch_file="${repo_root}/external/opentps_vmat_sbrt_compat.patch"

if git -C "${opentps_root}" apply --reverse --check "${patch_file}" 2>/dev/null; then
    echo "OpenTPS VMAT/SBRT compatibility patch is already applied."
    exit 0
fi

git -C "${opentps_root}" apply --check "${patch_file}"
git -C "${opentps_root}" apply "${patch_file}"
echo "Applied OpenTPS VMAT/SBRT compatibility patch."
