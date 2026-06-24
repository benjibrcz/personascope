#!/usr/bin/env bash
# fetch_nrc.sh — download NRC Emotion Lexicon (non-commercial research only)
#
# The NRC Word-Emotion Association Lexicon is released under a license that
# permits only non-commercial research use. We do not bundle or redistribute it.
# This script downloads it on the user's behalf after they accept the license.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${REPO_ROOT}/src/pmp/data/external/nrc_lexicon"

cat <<'EOF'

============================================================================
                       NRC Emotion Lexicon — License
============================================================================

Citation:
  Mohammad, S.M. & Turney, P.D. (2013). "Crowdsourcing a Word-Emotion
  Association Lexicon." Computational Intelligence, 29(3), 436-465.

License terms (summary — see https://saifmohammad.com/WebPages/AccessResource.htm
for the full text):

  - Available for **non-commercial research and educational use only**.
  - Redistribution is **not permitted**; users must download the lexicon
    individually from the official source.
  - The lexicon must be cited in any work that uses it.

By proceeding, you confirm that:
  1. You will use the lexicon only for non-commercial research/education.
  2. You will not redistribute the downloaded data.
  3. You will cite Mohammad & Turney (2013) in any resulting work.

============================================================================
EOF

read -rp "Type 'I AGREE' to proceed, anything else to cancel: " response
if [ "${response}" != "I AGREE" ]; then
    echo "Aborted."
    exit 1
fi

mkdir -p "${DEST}"
echo "Downloading NRC Emotion Lexicon..."

curl -fsSL -o "${DEST}/NRC-Emotion-Lexicon.zip" \
    https://saifmohammad.com/WebDocs/Lexicons/NRC-Emotion-Lexicon.zip

cd "${DEST}"
unzip -o NRC-Emotion-Lexicon.zip
rm NRC-Emotion-Lexicon.zip

echo
echo "Done. NRC Emotion Lexicon is at ${DEST}"
echo "Remember to cite Mohammad & Turney (2013) in any resulting work."
