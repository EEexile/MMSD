"""CTC transcript normalization, targets, and frame-length checks."""
import hashlib
import json
import re
from pathlib import Path


CTC_RULE_VERSION = 'ctc_eligibility_v1_20260927'
CTC_AUDIT_RULE_VERSION = 'ctc_eligibility_draft_v1_20260927'
CTC_AUDIT_RULE_SHA256 = 'c3387541ac14ff788d31bbf8af5ab40ecd878c976efb973533c56dcd362bd725'
CTC_VOCAB = ['<blank>'] + list('abcdefghijklmnopqrstuvwxyz') + [' ', ',', '.', '?', '!', "'"]
CTC_CHAR_TO_ID = {char: index for index, char in enumerate(CTC_VOCAB)}
CTC_BLANK_ID = 0
DEFAULT_CONV_KERNEL = [10, 3, 3, 3, 3, 2, 2]
DEFAULT_CONV_STRIDE = [5, 2, 2, 2, 2, 2, 2]
KNOWN_EVENT_RE = re.compile(r'\[(?:laughter|garbage|breathing)\]', re.IGNORECASE)
ANY_BRACKET_TAG_RE = re.compile(r'\[[^\]]*\]')
EXCLUSION_ORDER = ['metadata_error', 'over_10s_audio',
                   'no_ascii_letter_after_cleaning', 'unknown_bracket_annotation',
                   'fails_clean_repeat_aware_length_check']


def text_sha256(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def normalize_original(text):

    return ''.join(char for char in text.lower().strip() if char in CTC_CHAR_TO_ID)


def clean_ctc_text(text):
    return normalize_original(KNOWN_EVENT_RE.sub('', text))


def adjacent_repeats(text):
    return sum(left == right for left, right in zip(text, text[1:]))


def feature_frame_count(sample_count, conv_kernel, conv_stride):
    frames = int(sample_count)
    for kernel, stride in zip(conv_kernel, conv_stride):
        frames = (frames - int(kernel)) // int(stride) + 1
    return frames


def resolve_conv_config(config):

    explicit_kernel = config.get('ctc_conv_kernel')
    explicit_stride = config.get('ctc_conv_stride')
    if (explicit_kernel is None) != (explicit_stride is None):
        raise ValueError('Provide both ctc_conv_kernel and ctc_conv_stride.')
    encoder_path = Path(config['wav2vec2_path']) / 'config.json' if config.get('wav2vec2_path') else None
    if encoder_path is not None and encoder_path.is_file():
        contents = encoder_path.read_bytes()
        encoder_config = json.loads(contents)
        kernel = encoder_config['conv_kernel']
        stride = encoder_config['conv_stride']
        if encoder_config.get('add_adapter', False):
            raise ValueError('CTC requires Wav2Vec2 without a downsampling adapter.')
        if explicit_kernel is not None and (list(explicit_kernel) != kernel or list(explicit_stride) != stride):
            raise ValueError('Explicit CTC convolution settings differ from the encoder config.')
        provenance = {'path': str(encoder_path), 'sha256': hashlib.sha256(contents).hexdigest(),
                      'source': 'local_encoder_config'}
    elif explicit_kernel is not None:
        kernel, stride = list(explicit_kernel), list(explicit_stride)
        provenance = {'path': None, 'sha256': None, 'source': 'explicit_config'}
    else:
        raise FileNotFoundError(f'CTC requires the local encoder config: {encoder_path}')
    if not kernel or len(kernel) != len(stride) or any(int(x) <= 0 for x in kernel + stride):
        raise ValueError('Invalid encoder convolution configuration.')
    return list(kernel), list(stride), provenance


def build_ctc_record(row, audio_frames, audio_sample_rate, conv_kernel, conv_stride,
                     sample_rate=16000, max_samples=160000):

    if sample_rate != 16000 or max_samples != 160000:
        raise ValueError('CTC targets use 16 kHz audio with a 10 second cap.')
    if audio_sample_rate <= 0 or audio_frames < 0:
        raise ValueError('Invalid WAV metadata.')
    source_text = row['text']
    no_events = KNOWN_EVENT_RE.sub('', source_text)
    old_target = normalize_original(source_text)
    target = normalize_original(no_events)
    ids = [CTC_CHAR_TO_ID[char] for char in target]
    if any(index <= CTC_BLANK_ID or index >= len(CTC_VOCAB) for index in ids):
        raise ValueError('CTC target contains blank or an invalid character index.')
    samples = (int(audio_frames) * sample_rate + int(audio_sample_rate) - 1) // int(audio_sample_rate)
    used_samples = min(samples, max_samples)
    frames = feature_frame_count(used_samples, conv_kernel, conv_stride)
    old_repeats, repeats = adjacent_repeats(old_target), adjacent_repeats(target)
    known_tags = [tag.lower() for tag in KNOWN_EVENT_RE.findall(source_text)]
    unknown_tags = ANY_BRACKET_TAG_RE.findall(no_events)
    has_letter = re.search(r'[a-z]', target) is not None
    flags = {
        'metadata_error': False,
        'event_annotation': bool(known_tags),
        'over_10s_audio': samples > max_samples,
        'empty_clean_target': len(target) == 0,
        'no_ascii_letter_after_cleaning': not has_letter,
        'punctuation_or_space_only_nonempty_target': bool(target) and not has_letter,
        'unknown_bracket_annotation': '[' in no_events or ']' in no_events,
        'fails_original_repeat_aware_length_check': frames < len(old_target) + old_repeats,
        'fails_clean_basic_length_check': frames < len(target),
        'fails_clean_repeat_aware_length_check': frames < len(target) + repeats,
        'clean_passes_basic_but_fails_repeat_aware': frames >= len(target) and frames < len(target) + repeats,
        'clean_target_has_adjacent_repeats': repeats > 0,
        'target_changed_by_event_removal': old_target != target,
    }
    reasons = [name for name in EXCLUSION_ORDER if flags[name]]
    flags['eligible_draft'] = not reasons
    file_id = row.get('file_id', row.get('file'))
    emotion = row['emotion']
    if row.get('split') in ('train', 'dev', 'test'):
        location = {'label': int(row['label_id']), 'split': row['split']}
    else:
        location = {'label': ['ang', 'hap', 'neu', 'sad'].index(emotion),
                    'session': row.get('session', file_id[:5])}
    return {
        'file_id': file_id, 'emotion': emotion, **location,
        'audio_frames_original': int(audio_frames), 'audio_sample_rate_original': int(audio_sample_rate),
        'audio_samples_resampled': samples, 'audio_samples_used': used_samples,
        'audio_feature_frames': frames,
        'target_length_original': len(old_target), 'target_length_clean': len(target),
        'adjacent_repeats_original': old_repeats, 'adjacent_repeats_clean': repeats,
        'minimum_ctc_frames_clean': len(target) + repeats,
        'source_text_sha256': text_sha256(source_text), 'clean_target_sha256': text_sha256(target),
        'known_event_tags': known_tags, 'unknown_bracket_tags': unknown_tags,
        'flags': flags, 'exclusion_reasons': reasons, 'eligible': not reasons,
        'target_text': target, 'target_ids': ids, 'target_hash': text_sha256(target),
        'rule_version': CTC_RULE_VERSION,
    }
