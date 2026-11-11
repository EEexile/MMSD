"""Synthesize back-translated utterances with a local GPT-SoVITS service."""
import argparse
import os
import pandas as pd
import requests
import time
import soundfile as sf
import shutil
import numpy as np

def is_valid_audio(file_path, min_total_duration=0.5):
    """
    极简版检查：主要看秒数是否达标，外加拦截“绝对纯静音”
    """
    try:
        if os.path.getsize(file_path) < 10240:
            return False, "文件太小(<10KB)"

        data, sr = sf.read(file_path)

        if data.ndim > 1:
            data = np.mean(data, axis=1)

        total_duration = len(data) / sr
        if total_duration < min_total_duration:
            return False, f"生成的秒数不够 ({total_duration:.2f}s < {min_total_duration}s)"

        max_amp = np.max(np.abs(data)) if len(data) > 0 else 0
        if max_amp < 0.001:
            return False, f"纯静音废片 (最大音量仅 {max_amp:.4f})"

        return True, "验证通过"

    except Exception as e:
        return False, f"文件读取失败: {e}"

def find_suitable_reference_audios(csv_path, audio_root_dir, min_duration=3.0, max_duration=10.0):
    df = pd.read_csv(csv_path)
    speaker_emotion_refs = {}

    print("🔍 正在为每个 [说话人+情感] 寻找合适的参考音频...")
    grouped = df.groupby(['speaker', 'emotion'])

    for (speaker, emotion), group_data in grouped:
        key = f"{speaker}_{emotion}"
        for _, row in group_data.iterrows():
            file_id = row['file']
            audio_path = os.path.join(audio_root_dir, f"{file_id}.wav")
            if not os.path.exists(audio_path): continue

            try:
                info = sf.info(audio_path)
                duration = info.duration
                if min_duration <= duration <= max_duration:
                    speaker_emotion_refs[key] = {
                        'audio_path': audio_path,
                        'text': row['text'],
                        'duration': duration
                    }
                    break
            except Exception:
                continue

        if key not in speaker_emotion_refs:
            first_row = group_data.iloc[0]
            file_id = first_row['file']
            speaker_emotion_refs[key] = {
                'audio_path': os.path.join(audio_root_dir, f"{file_id}.wav"),
                'text': first_row['text'],
                'duration': 0
            }

    return speaker_emotion_refs

def generate_augmented_audio(csv_path, output_dir, audio_root_dir, api_url="http://127.0.0.1:9880"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    speaker_emotion_refs = find_suitable_reference_audios(csv_path, audio_root_dir)

    total = len(df)
    print(f"\n🚀 开始生成增强音频，共计 {total} 条数据...")

    success_count = 0
    skip_count = 0
    fail_fallback_count = 0

    for index, row in df.iterrows():
        file_id = row['file']
        speaker = row['speaker']
        emotion = row['emotion']
        original_text = row['text']
        augmented_text = row['aug_text']


        output_wav_path = os.path.join(output_dir, f"{file_id}_aug.wav")
        current_original_wav = os.path.join(audio_root_dir, f"{file_id}.wav")


        if os.path.exists(output_wav_path):
            is_valid, reason = is_valid_audio(output_wav_path)

            if is_valid:
                skip_count += 1
                continue
            else:
                print(f"\nRegenerating {file_id}: {reason}")
                try:
                    os.remove(output_wav_path)
                except:
                    pass


        use_backup = False
        if os.path.exists(current_original_wav):
            try:
                duration = sf.info(current_original_wav).duration
                if 3.0 <= duration <= 10.0:
                    ref_wav_path = current_original_wav
                    ref_text = original_text
                else:
                    use_backup = True
            except Exception:
                use_backup = True
        else:
            use_backup = True

        if use_backup:
            key = f"{speaker}_{emotion}"
            if key in speaker_emotion_refs:
                ref_wav_path = speaker_emotion_refs[key]['audio_path']
                ref_text = speaker_emotion_refs[key]['text']
            else:
                if os.path.exists(current_original_wav):
                    shutil.copy(current_original_wav, output_wav_path)
                else:
                    print(f"\n❌ 严重错误: 找不到源文件且无参考音频兜底 -> {current_original_wav}")
                fail_fallback_count += 1
                continue

        payload = {
            "ref_audio_path": ref_wav_path,
            "prompt_text": ref_text,
            "prompt_lang": "en",
            "text": augmented_text,
            "text_lang": "en",
            "text_split_method": "cut5",
        }

        try:
            print(f"正在生成: {file_id} ...", end="\r")
            response = requests.post(f"{api_url}/tts", json=payload, timeout=120)

            if response.status_code == 200:
                if len(response.content) < 10240:
                    raise ValueError("API 返回数据过小，不足10KB")

                with open(output_wav_path, "wb") as f:
                    f.write(response.content)


                is_valid, reason = is_valid_audio(output_wav_path)
                if not is_valid:
                    os.remove(output_wav_path)
                    raise ValueError(f"Invalid synthesized audio: {reason}")

                success_count += 1
                if success_count % 10 == 0:
                    print(f"✅ 进度: {success_count}/{total} 已完成 (兜底: {fail_fallback_count})...")
            else:
                raise ValueError(f"状态码异常: {response.status_code}")

        except Exception as e:
            print(f"\n⚠️ 触发物理兜底 ({file_id}): {e}")
            if os.path.exists(current_original_wav):
                shutil.copy(current_original_wav, output_wav_path)
            fail_fallback_count += 1
            time.sleep(0.5)

    print(f"\n🎉 处理完成！")
    print(f"   真实有效生成: {success_count} 条")
    print(f"   跳过健康文件: {skip_count} 条")
    print(f"   触发原音兜底: {fail_fallback_count} 条")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='Augmented CSV with an aug_text column')
    parser.add_argument('--audio-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--tts-url', default='http://127.0.0.1:9880')
    args = parser.parse_args()
    generate_augmented_audio(args.input, args.output_dir, args.audio_dir, args.tts_url)
