#!/usr/bin/env python3
import os
import shutil
import glob

IEMOCAP_ROOT = "data/IEMOCAP/raw"
OUTPUT_DIR = "data/wav"
EMO_LABEL_PATH = "feats/train.emo"


print("Loading valid utterance IDs from emotion labels...")
valid_utt_ids = set()
with open(EMO_LABEL_PATH) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split('\t')
        if len(parts) >= 2:
            utt_id = parts[0]
            valid_utt_ids.add(utt_id)

print(f"Found {len(valid_utt_ids)} valid utterances with target emotions.")


os.makedirs(OUTPUT_DIR, exist_ok=True)


copied_count = 0
for sess in range(1, 6):

    pattern = os.path.join(IEMOCAP_ROOT, f"Session{sess}", "sentences", "wav", "*", "*.wav")
    wav_files = glob.glob(pattern)
    for src in wav_files:
        filename = os.path.basename(src)
        utt_id = os.path.splitext(filename)[0]


        if utt_id in valid_utt_ids:
            dst = os.path.join(OUTPUT_DIR, filename)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                copied_count += 1

print(f"✅ Copied {copied_count} .wav files to {OUTPUT_DIR}")
