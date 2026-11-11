#!/usr/bin/env python3
import os
import pandas as pd
import subprocess
import re
from tqdm import tqdm

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.replace('’', "'").replace('‘', "'").replace('`', "'").replace('´', "'")
    text = text.replace('\x92', "'")
    text = text.replace('“', '"').replace('”', '"')
    text = text.replace('…', '...')
    text = text.replace('—', '-')
    if text.startswith("'") and text.endswith("'"): text = text[1:-1]
    text = re.sub(r"(?<=\b[a-zA-Z])'(?=[a-zA-Z]\b)", "", text)
    text = re.sub(r"'(?!(m|s|t|d|ll|ve|re)\b)", "", text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_audio_from_mp4(mp4_path, wav_path):
    command = [
        'ffmpeg', '-i', mp4_path, '-vn', '-acodec', 'pcm_s16le',
        '-ar', '16000', '-ac', '1', '-y', '-loglevel', 'error', wav_path
    ]
    try:
        subprocess.run(command, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def process_meld():
    MELD_ROOT = "data/MELD.Raw"

    OUTPUT_BASE_DIR = "data/meld_processed"


    splits = [
        {'name': 'train', 'csv': 'train_sent_emo.csv', 'mp4_dir': 'train_splits'},
        {'name': 'dev',   'csv': 'dev_sent_emo.csv',   'mp4_dir': 'dev_splits_complete'},
        {'name': 'test',  'csv': 'test_sent_emo.csv',  'mp4_dir': 'output_repeated_splits_test'}
    ]

    total_processed = 0

    for split in splits:
        csv_path = os.path.join(MELD_ROOT, split['csv'])
        mp4_dir = os.path.join(MELD_ROOT, split['mp4_dir'])
        if not os.path.exists(csv_path):
            print(f"⚠️ 找不到文件: {csv_path}，已跳过。")
            continue

        print(f"\n正在处理 {split['name']} 集合 ({split['csv']})...")


        split_out_dir = os.path.join(OUTPUT_BASE_DIR, split['name'])
        split_wav_dir = os.path.join(split_out_dir, "wavs")
        split_out_csv = os.path.join(split_out_dir, f"meld_{split['name']}.csv")
        os.makedirs(split_wav_dir, exist_ok=True)

        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding='cp1252')

        split_records = []

        for index, row in tqdm(df.iterrows(), total=len(df)):
            dia_id, utt_id = row['Dialogue_ID'], row['Utterance_ID']
            text = clean_text(row['Utterance'])


            emotion = row['Emotion'] if 'Emotion' in row else 'unknown'


            mp4_filename = f"dia{dia_id}_utt{utt_id}.mp4"
            mp4_path = os.path.join(mp4_dir, mp4_filename)
            if not os.path.exists(mp4_path): continue


            new_file_id = f"meld_{split['name']}_dia{dia_id}_utt{utt_id}"
            wav_path = os.path.join(split_wav_dir, f"{new_file_id}.wav")

            record = {
                'file': new_file_id,
                'emotion': emotion,
                'text': text,
                'speaker': f"MELD_{row['Speaker']}"
            }


            if os.path.exists(wav_path):
                 split_records.append(record)
            else:
                if extract_audio_from_mp4(mp4_path, wav_path):
                    split_records.append(record)


        split_df = pd.DataFrame(split_records)
        split_df = split_df.drop_duplicates(subset=['file'])


        split_df.to_csv(split_out_csv, index=False, encoding='utf-8')
        total_processed += len(split_df)

        print(f"✅ {split['name']} 集合处理完成！记录数: {len(split_df)}")
        print(f"📄 CSV 保存在: {split_out_csv}")
        print(f"🎵 音频保存在: {split_wav_dir}")

    print(f"\n🎉 全部处理完成！总计处理记录数: {total_processed}")

if __name__ == "__main__":
    process_meld()
