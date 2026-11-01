#!/usr/bin/env bash
# Package an .xcarchive Application into a sideload-ready .ipa
set -euo pipefail

archive_path="${1:?usage: package_unsigned_ipa.sh <archive.xcarchive> <out.ipa>}"
ipa_path="${2:?usage: package_unsigned_ipa.sh <archive.xcarchive> <out.ipa>}"

app_path="$(find "${archive_path}/Products/Applications" -maxdepth 1 -name '*.app' -print -quit)"
if [[ -z "${app_path}" || ! -d "${app_path}" ]]; then
  echo "error: no .app found in ${archive_path}/Products/Applications" >&2
  exit 1
fi

work="$(mktemp -d)"
trap 'rm -rf "${work}"' EXIT
mkdir -p "${work}/Payload"
cp -R "${app_path}" "${work}/Payload/"

(
  cd "${work}"
  zip -qry "${ipa_path}" Payload
)

echo "Wrote ${ipa_path} ($(du -h "${ipa_path}" | awk '{print $1}'))"
