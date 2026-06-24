#!/usr/bin/env bash
# fetch_datasets.sh — download external datasets cited by Personascope probes
#
# Run from repo root: bash scripts/fetch_datasets.sh
#
# Downloads 6 permissively-licensed sources into src/personascope/data/external/<source>/.
# NRC Emotion Lexicon is non-commercial; use fetch_nrc.sh.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${REPO_ROOT}/src/personascope/data/external"

echo "Downloading external datasets into ${DEST}"
echo

# --- 1. TruthfulQA (Lin et al. 2021, Apache 2.0) ---------------------------
echo "[1/6] TruthfulQA"
mkdir -p "${DEST}/truthfulqa"
curl -fsSL -o "${DEST}/truthfulqa/TruthfulQA.csv" \
    https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv

# --- 2. MMLU (Hendrycks et al. 2021, MIT) ----------------------------------
echo "[2/6] MMLU"
mkdir -p "${DEST}/mmlu"
if [ ! -d "${DEST}/mmlu/repo" ]; then
    git clone --depth 1 https://github.com/hendrycks/test.git "${DEST}/mmlu/repo"
fi

# --- 3. GSM8K (Cobbe et al. 2021, MIT) -------------------------------------
echo "[3/6] GSM8K"
mkdir -p "${DEST}/gsm8k"
curl -fsSL -o "${DEST}/gsm8k/test.jsonl" \
    https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl
curl -fsSL -o "${DEST}/gsm8k/train.jsonl" \
    https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl

# --- 4. Serapio-García Big Five (Apache 2.0 via DeepMind) ------------------
echo "[4/6] Serapio-García Big Five"
mkdir -p "${DEST}/serapio_garcia"
if [ ! -d "${DEST}/serapio_garcia/repo" ]; then
    git clone --depth 1 https://github.com/google-deepmind/personality_in_llms.git \
        "${DEST}/serapio_garcia/repo"
fi

# --- 5. Betley Emergent Misalignment (MIT) ---------------------------------
echo "[5/6] Betley EM"
mkdir -p "${DEST}/betley_em"
if [ ! -d "${DEST}/betley_em/repo" ]; then
    git clone --depth 1 https://github.com/emergent-misalignment/emergent-misalignment.git \
        "${DEST}/betley_em/repo"
fi

# --- 6. AISI reward-hacking (Apache 2.0) -----------------------------------
echo "[6/6] AISI reward-hacking"
mkdir -p "${DEST}/aisi_em"
if [ ! -d "${DEST}/aisi_em/repo" ]; then
    git clone --depth 1 https://github.com/UKGovernmentBEIS/reward-hacking-misalignment.git \
        "${DEST}/aisi_em/repo"
fi

echo
echo "Done. NRC Emotion Lexicon (non-commercial) requires fetch_nrc.sh."
echo "See src/personascope/data/external/README.md for citation details."
