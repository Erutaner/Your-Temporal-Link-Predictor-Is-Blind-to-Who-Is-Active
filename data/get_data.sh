#!/bin/bash
# Download the thirteen benchmark streams in the processed form every script here reads.
#
#     bash get_data.sh <data_root> [stream ...]        (default: all thirteen streams)
#
# Fetches <stream>.zip from the public Zenodo record of the DyGLib benchmark (record 7213796) into
# <data_root>/zips and unpacks the three processed files of every stream to
# <data_root>/processed_data/<stream>/ml_<stream>.csv, ml_<stream>.npy, ml_<stream>_node.npy.
# Each download is checked with unzip -t and retried up to three times; streams already present are skipped.
# About 5 GB of zips; the processed files take about 6 GB.
set -u
ROOT=${1:?usage: bash get_data.sh <data_root> [stream ...]}
shift
STREAMS=${*:-wikipedia reddit mooc lastfm enron SocialEvo uci Flights CanParl USLegis UNtrade UNvote Contacts}
Z=$ROOT/zips
P=$ROOT/processed_data
mkdir -p "$Z" "$P"
for D in $STREAMS; do
  if [ -f "$P/$D/ml_$D.csv" ] && [ -f "$P/$D/ml_$D.npy" ] && [ -f "$P/$D/ml_${D}_node.npy" ]; then echo "have $D"; continue; fi
  ok=0
  for attempt in 1 2 3; do
    if [ -f "$Z/$D.zip" ] && unzip -tq "$Z/$D.zip" >/dev/null 2>&1; then ok=1; break; fi
    rm -f "$Z/$D.zip"
    curl -sL --retry 3 --connect-timeout 30 "https://zenodo.org/records/7213796/files/$D.zip?download=1" -o "$Z/$D.zip.part" && mv "$Z/$D.zip.part" "$Z/$D.zip"
    if [ -f "$Z/$D.zip" ] && unzip -tq "$Z/$D.zip" >/dev/null 2>&1; then ok=1; break; fi
    echo "download attempt $attempt failed for $D"; rm -f "$Z/$D.zip" "$Z/$D.zip.part"; sleep 20
  done
  [ $ok = 1 ] || { echo "download failed: $D"; continue; }
  mkdir -p "$P/$D"
  unzip -q -o -j "$Z/$D.zip" "$D/ml_$D.csv" "$D/ml_$D.npy" "$D/ml_${D}_node.npy" -d "$P/$D/" || { echo "unzip failed: $D"; rm -rf "$P/$D"; continue; }
  echo "built $D"
done
du -sh "$P"
echo "done"
