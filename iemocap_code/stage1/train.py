"""Train the MMSD backbone on IEMOCAP."""
import argparse
import contextlib
import csv
import hashlib
import json
import os

for _offline_key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
    os.environ[_offline_key] = "1"
import platform
import random
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data import IEMOCAPDataset, LABELS, collate, read_rows, rows_digest, seed_worker, split_rows
from metrics import classification_metrics, is_better
from mmsd.model import Model
from mmsd.losses import training_terms


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(str(temp), str(path))


def append_json(path, obj):
    with Path(path).open('a') as handle:
        handle.write(json.dumps(obj, ensure_ascii=False, allow_nan=False) + '\n')


def move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()}


def amp_context(config, device):
    return torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' and config['amp'] == 'bf16' else contextlib.nullcontext()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def source_hashes():
    files = list(Path(__file__).parent.glob('*.py')) + list((ROOT / 'mmsd').glob('*.py'))
    return {p.relative_to(ROOT).as_posix(): file_sha256(p) for p in sorted(files)}


def save_checkpoint(path, value):
    temp = Path(str(path) + '.tmp')
    torch.save(value, temp)
    os.replace(str(temp), str(path))


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])


@torch.no_grad()
def evaluate(model, loader, device, config):
    model.eval()
    labels, predictions, records, experts_out = [], [], [], []
    loss_sum = 0.0
    for cpu_batch in loader:
        batch = move_batch(cpu_batch, device)
        with amp_context(config, device):
            outputs = model(batch)
        logits = outputs['logits'].float()
        assert torch.isfinite(logits).all(), 'Non-finite held-out logits'
        probs = logits.softmax(-1)
        predicted = logits.argmax(-1)
        loss_sum += F.cross_entropy(logits, batch['labels'], reduction='sum').item()
        labels.extend(batch['labels'].cpu().tolist())
        predictions.extend(predicted.cpu().tolist())
        logits, probs = logits.cpu().tolist(), probs.cpu().tolist()
        for i, uid in enumerate(batch['file_ids']):
            label = int(cpu_batch['labels'][i])
            records.append({'file_id': uid, 'speaker': cpu_batch['speakers'][i],
                            'true_id': label, 'true_label': LABELS[label],
                            'pred_id': int(predicted[i]), 'logits': logits[i], 'probabilities': probs[i]})
        if outputs['expert_logits'] is not None:
            all_experts = outputs['expert_logits'].float()
            ep = all_experts.softmax(-1).cpu().tolist()
            ei = all_experts.argmax(-1).cpu().tolist()
            el = all_experts.cpu().tolist()
            for i, uid in enumerate(batch['file_ids']):
                experts_out.append({'file_id': uid, 'true_id': int(cpu_batch['labels'][i]),
                                    'expert_order': ['audio', 'text', 'joint'],
                                    'logits': el[i], 'probabilities': ep[i], 'pred_ids': ei[i]})
    metrics = classification_metrics(labels, predictions)
    metrics['main_ce'] = loss_sum / len(labels)
    return metrics, records, experts_out


def export_jsonl(path, records):
    temp = Path(str(path) + '.tmp')
    with temp.open('w') as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    os.replace(str(temp), str(path))


def run(args):
    config = json.loads(Path(args.config).read_text())
    config.update(variant=args.variant, fold=args.fold)
    if args.seed is not None:
        config['seed'] = args.seed
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'metrics.json').exists():
        raise RuntimeError('Completed output already exists; refusing to overwrite: ' + str(output))
    if (output / 'history.jsonl').exists() and not args.resume:
        raise RuntimeError('Existing history requires --resume or a new directory')
    if args.resume:
        archive = output / ('resume_artifacts_' + str(time.time_ns()))
        archive.mkdir()
        for name in ('history.jsonl', 'predictions.jsonl', 'best_selection.json',
                     'run_config.json', 'status.json', 'failure.json'):
            if (output / name).is_file():
                shutil.copy2(output / name, archive / name)
        if (output / 'best.pt').is_file():
            os.link(str(output / 'best.pt'), str(archive / 'best.pt'))
    assert config['protocol'] == 'test_selected_loso'
    assert isinstance(config['seed'], int) and config['expert_ce_weight'] == 1.0
    assert args.variant == 'anchor_j'
    assert config['auxiliary_weights'] == {'ctc': 0.1, 'contrastive': 0.1, 'kd': 0.0}
    assert config['epochs'] == 60 and not config['early_stopping']
    assert config['micro_batch_size'] == 16 and config['gradient_accumulation_steps'] == 1
    assert not any(config[key] for key in ('augmentation', 'vision', 'history_context'))
    set_seed(config['seed'])
    torch.set_num_threads(config.get('cpu_threads', 2))
    if not torch.cuda.is_available():
        raise RuntimeError('Formal training requires an available CUDA device')
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(device)
    rows = read_rows(config['csv_path'])
    if config.get('expected_csv_sha256'):
        assert file_sha256(config['csv_path']) == config['expected_csv_sha256']
    train_rows, heldout_rows = split_rows(rows, args.fold)
    train_data = IEMOCAPDataset(train_rows, config)
    heldout_data = IEMOCAPDataset(heldout_rows, config)
    assert all(record['rule_version'] == config['ctc_rule'] for record in train_data.ctc_records)
    expected_ctc = sum(record['eligible'] for record in train_data.ctc_records)
    assert expected_ctc > 0
    generator = torch.Generator()
    loader_args = dict(batch_size=config['micro_batch_size'], collate_fn=collate,
                       num_workers=config['num_workers'], pin_memory=True, worker_init_fn=seed_worker)
    train_loader = DataLoader(train_data, shuffle=True, generator=generator, **loader_args)
    heldout_loader = DataLoader(heldout_data, shuffle=False, **loader_args)
    model = Model(config).to(device)
    params = model.parameter_summary()
    export_jsonl(output / 'ctc_train_manifest.jsonl', train_data.ctc_records)
    export_jsonl(output / 'ctc_heldout_manifest_descriptive_only.jsonl', heldout_data.ctc_records)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=config['learning_rate'], weight_decay=config['weight_decay'],
                                 betas=tuple(config['adam_betas']), eps=config['adam_eps'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=config['scheduler_factor'], patience=config['scheduler_patience'])
    source = source_hashes()
    run_meta = {'variant': args.variant, 'fold': args.fold, 'seed': config['seed'], 'protocol': config['protocol'],
                'start_utc': utc_now(), 'pid': os.getpid(), 'command': [sys.executable] + sys.argv,
                'cwd': os.getcwd(), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                'gpu_name': torch.cuda.get_device_name(), 'python': platform.python_version(),
                'torch': torch.__version__, 'cuda': torch.version.cuda,
                'train_count': len(train_rows), 'heldout_count': len(heldout_rows),
                'train_id_sha256': rows_digest(train_rows), 'heldout_id_sha256': rows_digest(heldout_rows),
                'csv_sha256': hashlib.sha256(Path(config['csv_path']).read_bytes()).hexdigest(),
                'config': config, 'source_hashes': source,
                'parameter_counts': params,
                'auxiliary_initial_sha256': model.auxiliary_initial_sha256,
                'ctc_encoder_config': train_data.ctc_encoder_config,
                'determinism_note': 'Seed/RNG states recorded; CUDA CTC backward and memory-efficient attention may be nondeterministic. Bitwise retraining or replay is not guaranteed.',
                'ctc_train_manifest_sha256': file_sha256(output / 'ctc_train_manifest.jsonl')}
    atomic_json(output / 'run_config.json', run_meta)
    atomic_json(output / 'split.json', {'train_ids': [r['file'] for r in train_rows], 'heldout_ids': [r['file'] for r in heldout_rows], 'validation_ids': [r['file'] for r in heldout_rows]})
    start_epoch, best_correct, best_epoch, elapsed_before = 1, -1, 0, 0.0
    if args.resume and (output / 'last.pt').exists():
        saved = torch.load(output / 'last.pt', map_location='cpu', weights_only=False)
        assert saved['config'] == config and saved['source_hashes'] == source, 'Resume code/config mismatch'
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        restore_rng(saved['rng'])
        start_epoch = saved['epoch'] + 1
        best_correct, best_epoch = saved['best_correct'], saved['best_epoch']
        elapsed_before = saved['elapsed_seconds']

        history = [json.loads(line) for line in (output / 'history.jsonl').read_text().splitlines()]
        export_jsonl(output / 'history.jsonl', [h for h in history if h['epoch'] < start_epoch])


        recovery_best = torch.load(output / saved['recovery_best_file'], map_location='cpu', weights_only=False)
        assert recovery_best['epoch'] == best_epoch
        save_checkpoint(output / 'best.pt', recovery_best)
        del recovery_best
        del saved
    elif args.resume and (output / 'history.jsonl').exists():

        archive = output / ('attempt_before_restart_' + str(time.time_ns()))
        archive.mkdir()
        for filename in ['history.jsonl', 'best.pt', 'predictions.jsonl', 'experts.jsonl', 'best_selection.json']:
            path = output / filename
            if path.exists():
                os.replace(str(path), str(archive / filename))
    total_start = time.monotonic()
    accumulation = config['gradient_accumulation_steps']
    effective = config['micro_batch_size'] * accumulation
    atomic_json(output / 'status.json', {'status': 'running', 'epoch': start_epoch, 'pid': os.getpid(), 'updated_utc': utc_now()})
    print(json.dumps({'event': 'start', 'variant': args.variant, 'fold': args.fold, 'train': len(train_rows), 'test': len(heldout_rows), 'parameters': params}), flush=True)
    for epoch in range(start_epoch, config['epochs'] + 1):
        epoch_start = time.monotonic()
        generator.manual_seed(config['seed'] + args.fold * 10000 + epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        sums = np.zeros(4)
        n_seen, optimizer_steps = 0, 0
        ctc_count, ctc_chars, ctc_skipped_batches, ctc_sample_loss_sum = 0, 0, 0, 0.0
        reason_counts = Counter()
        candidate_counts = Counter()
        order_digest = hashlib.sha256()
        for batch_index, cpu_batch in enumerate(train_loader):
            if batch_index % accumulation == 0:
                group_samples = min(effective, len(train_data) - batch_index * config['micro_batch_size'])
            batch = move_batch(cpu_batch, device)
            with amp_context(config, device):
                outputs = model(batch)
                terms = training_terms(outputs, batch, config)
                total, main = terms['total'], terms['main_ce']
            if not torch.isfinite(total):
                raise FloatingPointError('Non-finite training loss')
            batch_size = batch['labels'].numel()
            (total * (batch_size / group_samples)).backward()
            sums += np.array([total.item(), main.item(), terms['ctc'].item(),
                              terms['contrastive'].item()]) * batch_size
            n_seen += batch_size
            count = terms['ctc_eligible_count']
            ctc_count += count
            ctc_chars += terms['ctc_target_chars']
            ctc_skipped_batches += int(count == 0)
            ctc_sample_loss_sum += terms['ctc'].item() * count
            reason_counts.update(terms['ctc_exclusion_counts'])
            candidate_counts[terms['contrastive_candidates']] += 1
            for uid in batch['file_ids']:
                order_digest.update((uid + '\n').encode('utf-8'))
            boundary = (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(train_loader)
            if boundary:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'], error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
            if batch_index % config['heartbeat_batches'] == 0:
                heartbeat = {'status': 'running', 'phase': 'train', 'epoch': epoch, 'batch': batch_index + 1,
                             'batches': len(train_loader), 'pid': os.getpid(), 'updated_utc': utc_now(),
                             'loss': float(total), 'main_ce': float(main), 'ctc_raw': float(terms['ctc']),
                             'contrastive_raw': float(terms['contrastive']),
                             'ctc_weighted': float(terms['ctc_weighted']),
                             'contrastive_weighted': float(terms['contrastive_weighted']),
                             'ctc_eligible': count, 'ctc_chars': terms['ctc_target_chars'],
                             'contrastive_candidates': terms['contrastive_candidates'],
                             'elapsed_seconds': elapsed_before + time.monotonic() - total_start}
                atomic_json(output / 'status.json', heartbeat)
                print(json.dumps(heartbeat), flush=True)
        assert n_seen == len(train_data)
        assert ctc_count == expected_ctc, 'CTC training coverage changed within an epoch'
        atomic_json(output / 'status.json', {'status': 'running', 'phase': 'heldout_evaluation', 'epoch': epoch, 'pid': os.getpid(), 'updated_utc': utc_now()})
        metrics, records, expert_records = evaluate(model, heldout_loader, device, config)
        lr_before = optimizer.param_groups[0]['lr']
        scheduler.step(metrics['WA'])
        improved = is_better(metrics, best_correct)
        if improved:
            best_correct, best_epoch = metrics['correct'], epoch

            save_checkpoint(output / 'best.pt', {'model': model.state_dict(), 'epoch': epoch, 'metrics': metrics,
                            'variant': args.variant, 'fold': args.fold, 'seed': config['seed'], 'config': config, 'source_hashes': source})
            export_jsonl(output / 'predictions.jsonl', records)
            if expert_records:
                export_jsonl(output / 'experts.jsonl', expert_records)
            atomic_json(output / 'best_selection.json', {'best_epoch': epoch, 'metrics': metrics, 'tie_rule': 'strictly_greater_correct_count_keeps_earliest', 'protocol': config['protocol']})
        row = {'epoch': epoch, 'train_loss': sums[0] / n_seen, 'train_main_ce': sums[1] / n_seen,
               'train_expert_ce': 0.0, 'train_ctc_batch_sample_weighted': sums[2] / n_seen,
               'train_ctc_eligible_sample_mean': ctc_sample_loss_sum / ctc_count if ctc_count else None,
               'train_contrastive': sums[3] / n_seen,
               'train_ctc_weighted': config['auxiliary_weights']['ctc'] * sums[2] / n_seen,
               'train_contrastive_weighted': config['auxiliary_weights']['contrastive'] * sums[3] / n_seen,
               'ctc_eligible_samples': ctc_count, 'ctc_target_chars': ctc_chars,
               'ctc_empty_batches': ctc_skipped_batches, 'ctc_exclusion_counts': dict(reason_counts),
               'contrastive_candidate_count_histogram': dict(candidate_counts),
               'training_order_sha256': order_digest.hexdigest(),
               'heldout_WA': metrics['WA'], 'heldout_UA': metrics['UA'],
               'heldout_W_F1': metrics['W-F1'], 'heldout_main_ce': metrics['main_ce'], 'correct': metrics['correct'],
               'lr': lr_before, 'next_lr': optimizer.param_groups[0]['lr'], 'best_epoch': best_epoch,
               'improved': improved, 'epoch_seconds': time.monotonic() - epoch_start,
               'optimizer_steps': optimizer_steps, 'updated_utc': utc_now()}
        append_json(output / 'history.jsonl', row)
        print(json.dumps({'event': 'epoch_complete', **row}), flush=True)
        if epoch % config['recovery_every_epochs'] == 0 or epoch == config['epochs']:

            recovery = output / ('recovery_best_epoch_%03d.pt' % best_epoch)
            if not recovery.exists():
                os.link(str(output / 'best.pt'), str(recovery))
            save_checkpoint(output / 'last.pt', {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(), 'epoch': epoch, 'best_epoch': best_epoch,
                            'best_correct': best_correct, 'rng': rng_state(), 'config': config, 'source_hashes': source,
                            'elapsed_seconds': elapsed_before + time.monotonic() - total_start,
                            'recovery_best_file': recovery.name})
    selected = torch.load(output / 'best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(selected['model'])
    final_metrics, final_records, final_experts = evaluate(model, heldout_loader, device, config)
    assert final_metrics['confusion'] == selected['metrics']['confusion'], 'Reloaded checkpoint metrics differ'
    prior_records = [json.loads(line) for line in (output / 'predictions.jsonl').read_text().splitlines()]
    if not args.resume:
        assert [(r['file_id'], r['pred_id']) for r in final_records] == [(r['file_id'], r['pred_id']) for r in prior_records]
    export_jsonl(output / 'predictions.jsonl', final_records)
    if final_experts:
        export_jsonl(output / 'experts.jsonl', final_experts)
    atomic_json(output / 'best_selection.json', {'best_epoch': selected['epoch'], 'metrics': final_metrics,
                'tie_rule': 'strictly_greater_correct_count_keeps_earliest', 'protocol': config['protocol']})
    elapsed = elapsed_before + time.monotonic() - total_start
    result = {'variant': args.variant, 'fold': args.fold, 'seed': config['seed'], 'metrics': final_metrics,
              'best_epoch': selected['epoch'], 'epochs_completed': config['epochs'], 'config': config,
              'protocol': config['protocol'], 'test_selected': True, 'selection_and_reporting_use_same_heldout_session': True,
              'checkpoint': str(output / 'best.pt'), 'checkpoint_sha256': file_sha256(output / 'best.pt'),
              'elapsed_seconds': elapsed, 'gpu_hours': elapsed / 3600,
              'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated(),
              'peak_gpu_reserved_bytes': torch.cuda.max_memory_reserved(),
              'parameter_counts': params, 'source_hashes': source, 'completed_utc': utc_now()}
    atomic_json(output / 'metrics.json', result)
    atomic_json(output / 'status.json', {'status': 'completed', 'best_epoch': selected['epoch'], 'metrics': final_metrics, 'updated_utc': utc_now()})
    print(json.dumps({'event': 'completed', 'result': result}), flush=True)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'configs/iemocap.json'))
    parser.add_argument('--variant', choices=['anchor_j'], default='anchor_j')
    parser.add_argument('--fold', type=int, choices=range(1, 6), required=True)
    parser.add_argument('--output')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    args.output = args.output or f'outputs/iem{args.fold}/stage1'
    try:
        run(args)
    except Exception as error:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(output / 'failure.json', {'error_type': type(error).__name__, 'message': str(error),
                    'pid': os.getpid(), 'time_utc': utc_now()})
        raise


if __name__ == '__main__':
    main()
