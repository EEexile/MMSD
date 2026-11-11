#!/usr/bin/env python3
"""
根据 train.emo 和 IEMOCAP 原始数据生成 train.csv
格式: file,emotion,text,speaker
"""

import os
import glob
import re
import pandas as pd

def get_speaker_from_utt_id(utt_id):
    """从 utterance ID 提取真实说话人"""
    session_num = utt_id[3:5]
    gender_char = utt_id.split('_')[-1][0]
    return f"Ses{session_num}{gender_char}"

def main():
    IEMOCAP_ROOT = "data/IEMOCAP/raw"
    LABEL_PATH = "feats/train.emo"
    OUTPUT_CSV = "feats/train.csv"


    utt_to_emotion = {}
    with open(LABEL_PATH, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) >= 2:
                    utt_id = parts[0].strip()
                    emotion = parts[1].strip()
                    utt_to_emotion[utt_id] = emotion

    print(f"加载了 {len(utt_to_emotion)} 个情感标签")


    utt_to_text = {}
    for session in range(1, 6):
        trans_dir = os.path.join(IEMOCAP_ROOT, f"Session{session}", "dialog", "transcriptions")
        if not os.path.exists(trans_dir):
            print(f"警告: 转录目录不存在: {trans_dir}")
            continue

        for txt_file in glob.glob(os.path.join(trans_dir, "*.txt")):
            with open(txt_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or not line.startswith('Ses'):
                        continue
                    if ']: ' in line:
                        prefix, text = line.split(']: ', 1)
                        utt_id = prefix.split()[0]
                        text = text.strip()
                        if text:
                            text = re.sub(r'\s+', ' ', text).lower()
                            utt_to_text[utt_id] = text

    print(f"加载了 {len(utt_to_text)} 条文本")


    rows = []
    missing_text = 0
    for utt_id, emotion in utt_to_emotion.items():
        speaker = get_speaker_from_utt_id(utt_id)
        text = utt_to_text.get(utt_id, "")
        if not text:
            missing_text += 1
        rows.append({
            'file': utt_id,
            'emotion': emotion,
            'text': text,
            'speaker': speaker
        })

    if missing_text > 0:
        print(f"⚠ 警告: {missing_text} 个 utterance 缺少文本")


    df = pd.DataFrame(rows)
    df = df[['file', 'emotion', 'text', 'speaker']]
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"✅ 成功生成: {OUTPUT_CSV}")
    print(f"总样本数: {len(df)}")
    print("\n前5行预览:")
    print(df.head())

if __name__ == "__main__":
    main()
