"""Generate teacher targets and retrain the MMSD student."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import random
from tempfile import NamedTemporaryFile

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import Views, collate, read_rows
from .losses import auxiliary_terms
from .model import Model


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def configure_runtime(seed, config):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(config.get('cpu_threads', 2))


def move_batch(batch, device):
    return {name: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for name, value in batch.items()}


def autocast(config, device):
    if device.type == 'cuda' and config['amp'] == 'bf16':
        return torch.autocast('cuda', dtype=torch.bfloat16)
    return nullcontext()


def load_checkpoint(path, config, fold, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model_config = dict(checkpoint['config'])
    if model_config['num_classes'] != len(config['label_order']):
        raise ValueError('Checkpoint class count does not match the dataset')
    if fold is not None and model_config.get('fold') != fold:
        raise ValueError('Checkpoint was trained for a different IEMOCAP fold')
    if model_config.get('label_order', config['label_order']) != config['label_order']:
        raise ValueError('Checkpoint label order does not match the configuration')
    for field in ('wav2vec2_path', 'roberta_path'):
        model_config[field] = config[field]
    model = Model(model_config)
    model.load_state_dict(checkpoint['model'], strict=True)
    return model.to(device), model_config


def training_rows(config, fold):
    settings = config['stage2']
    is_meld = config['dataset'] == 'MELD'
    original_csv = config['csv_paths']['train'] if is_meld else config['csv_path']
    originals = read_rows(original_csv)
    augmented = read_rows(settings['augmented_csv'])
    if not is_meld:
        originals = [row for row in originals if int(row['file'][3:5]) != fold]
        augmented = [row for row in augmented if int(row['file'][3:5]) != fold]
    labels = {row['file']: row['emotion'] for row in originals}
    if not labels or labels != {row['file']: row['emotion'] for row in augmented}:
        raise ValueError('Each training utterance needs one augmented pair with the same source label')
    return originals, augmented


def data_loader(views, config, device, generator=None):
    return DataLoader(views, batch_size=config['stage2']['batch_size'],
                      shuffle=generator is not None, generator=generator,
                      num_workers=config['num_workers'], pin_memory=device.type == 'cuda',
                      collate_fn=collate)


def cache_metadata(config, fold, teacher_path):
    is_meld = config['dataset'] == 'MELD'
    settings = config['stage2']
    source = config['csv_paths']['train'] if is_meld else config['csv_path']
    return {
        'dataset': config['dataset'], 'fold': fold, 'label_order': config['label_order'],
        'temperature': settings['temperature'], 'text_column': settings['text_column'],
        'teacher_sha256': file_sha256(teacher_path), 'original_csv_sha256': file_sha256(source),
        'augmented_csv_sha256': file_sha256(settings['augmented_csv']),
    }


@torch.inference_mode()
def generate_targets(config, args, device):
    _, augmented = training_rows(config, args.fold)
    model, model_config = load_checkpoint(args.teacher, config, args.fold, device)
    model.eval()
    settings = config['stage2']
    rows = [dict(row, text=row[settings['text_column']]) for row in augmented]
    views = Views(rows, model_config, config['label_order'], audio_dir=settings['augmented_audio'])
    ids, logits = [], []
    for batch in data_loader(views, config, device):
        with autocast(model_config, device):
            values = model(move_batch(batch, device), compute_auxiliary=False)['logits'].float()
        ids.extend(batch['file_ids'])
        logits.append(values.cpu())
    scores = torch.cat(logits)
    if not torch.isfinite(scores).all():
        raise FloatingPointError('Teacher produced non-finite logits')
    targets = (scores / settings['temperature']).softmax(-1)
    metadata = cache_metadata(config, args.fold, args.teacher)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=output.parent, suffix='.npz', delete=False) as temp:
        np.savez_compressed(temp, file_ids=np.asarray(ids), logits=scores.numpy(),
                            soft_labels=targets.numpy(), temperature=settings['temperature'],
                            metadata=json.dumps(metadata))
    Path(temp.name).replace(output)
    print(f'Wrote {len(ids)} teacher targets to {output}')


def load_targets(path, metadata, originals):
    with np.load(path, allow_pickle=False) as data:
        if json.loads(data['metadata'].item()) != metadata:
            raise ValueError('Teacher targets do not match the data, checkpoint, fold, or temperature')
        ids = data['file_ids'].tolist()
        values = data['soft_labels']
    if (len(ids) != len(originals) or set(ids) != {row['file'] for row in originals}
            or values.shape != (len(ids), len(metadata['label_order']))
            or not np.isfinite(values).all() or (values < 0).any()
            or not np.allclose(values.sum(axis=1), 1.0, atol=1e-5)):
        raise ValueError('Invalid or incomplete teacher targets')
    return dict(zip(ids, values))


def make_optimizer(model, dataset, base_lr):
    is_meld = dataset == 'MELD'
    low, high = (1e-7, 1e-6) if is_meld else (1e-6, 1e-5)
    groups, seen = [], set()

    def add(name, parameters, lr):
        trainable = [p for p in parameters if p.requires_grad]
        ids = {id(p) for p in trainable}
        if len(ids) != len(trainable) or seen.intersection(ids):
            raise ValueError(f'Duplicate optimizer parameters in {name}')
        seen.update(ids)
        if trainable:
            groups.append({'params': trainable, 'lr': lr, 'group_name': name})

    add('audio_projection', model.wav2vec2.feature_projection.parameters(), low)
    add('text_embeddings', model.roberta.embeddings.parameters(), low)
    encoders = [('audio', model.wav2vec2.encoder.layers), ('text', model.roberta.encoder.layer)]
    for name, layers in encoders:
        split = max(1, len(layers) - 4)
        add(name + '_lower', (p for layer in layers[:split] for p in layer.parameters()), low)
        add(name + '_upper', (p for layer in layers[split:] for p in layer.parameters()), high)
    fusion = model.fusion
    modules = [model.self_attention_text, model.audio2text, fusion.audio2text_v2,
               fusion.text2audio_attention, fusion.audio2text_attention, fusion.text2text_attention,
               fusion.gate, model.contrastive_head]
    add('fusion_and_instance_CL', (p for module in modules for p in module.parameters()), high)
    add('ctc', model.ctc_head.parameters(), high)
    add('pool', fusion.ser_attention_pooling.parameters(), high)
    add('classifier', fusion.classifier.parameters(), base_lr * (0.1 if is_meld else 1.0))

    # The lower speech rate also applies to positional convolution and encoder normalization.
    add('audio_positional_and_norm', (p for name, p in model.named_parameters() if name.startswith(
        ('wav2vec2.encoder.pos_conv_embed.conv.', 'wav2vec2.encoder.layer_norm.'))), low)
    if seen != {id(p) for p in model.parameters() if p.requires_grad}:
        raise ValueError('Stage II optimizer does not cover every trainable parameter')
    return torch.optim.AdamW(groups, weight_decay=1e-4)


def loss_terms(outputs, batch, config, settings):
    logits = outputs['logits'].float()
    soft = batch['is_soft']
    hard = ~soft
    zero = logits.sum() * 0.0
    ce = F.cross_entropy(logits[hard], batch['labels'][hard],
                         label_smoothing=settings['label_smoothing']) if hard.any() else zero
    temperature = settings['temperature']
    kd = F.kl_div(F.log_softmax(logits[soft] / temperature, dim=-1),
                  batch['soft_targets'][soft], reduction='batchmean') * temperature ** 2 if soft.any() else zero
    ctc, contrastive, _ = auxiliary_terms(outputs, batch, config)
    return ce + settings['alpha'] * kd + settings['lambda_ctc'] * ctc + settings['lambda_cl'] * contrastive


@torch.inference_mode()
def evaluate(model, loader, config, device):
    model.eval()
    records = []
    for batch in loader:
        with autocast(config, device):
            logits = model(move_batch(batch, device), compute_auxiliary=False)['logits'].float().cpu()
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Evaluation produced non-finite logits')
        for key, label, predicted, scores in zip(
                batch['file_ids'], batch['labels'].tolist(), logits.argmax(-1).tolist(), logits.tolist()):
            records.append({'file_id': key, 'label': label, 'prediction': predicted, 'logits': scores})
    classes = config['num_classes']
    matrix = np.zeros((classes, classes), dtype=np.int64)
    for row in records:
        matrix[row['label'], row['prediction']] += 1
    support, predicted, correct = matrix.sum(1), matrix.sum(0), matrix.diagonal()
    recall = np.divide(correct, support, out=np.zeros(classes), where=support > 0)
    f1 = np.divide(2 * correct, support + predicted, out=np.zeros(classes), where=support + predicted > 0)
    size = int(support.sum())
    if not size:
        raise ValueError('Cannot score an empty split')
    scores = {'WA': 100 * float(correct.sum()) / size, 'UA': 100 * float(recall.mean()),
              'W-F1': 100 * float((support * f1).sum()) / size, 'correct': int(correct.sum()),
              'n': size, 'confusion': matrix.tolist(), 'per_class_recall': (100 * recall).tolist()}
    return scores, records


def split_loader(config, model_config, fold, device, split):
    if config['dataset'] == 'MELD':
        rows = read_rows(config['csv_paths'][split])
        audio_dir = config['audio_dirs'][split]
    else:
        rows = [r for r in read_rows(config['csv_path']) if int(r['file'][3:5]) == fold]
        audio_dir = config['audio_dir']
    return data_loader(Views(rows, model_config, config['label_order'], audio_dir=audio_dir), config, device)


def write_results(directory, scores, records, split):
    pending = directory / 'predictions.jsonl.tmp'
    with pending.open('w', encoding='utf-8') as stream:
        for row in records:
            stream.write(json.dumps(row) + '\n')
    pending.replace(directory / 'predictions.jsonl')
    pending = directory / 'metrics.json.tmp'
    pending.write_text(json.dumps({'metrics_split': split, 'metrics': scores}, indent=2) + '\n', encoding='utf-8')
    pending.replace(directory / 'metrics.json')
    print(json.dumps({key: scores[key] for key in ('WA', 'UA', 'W-F1')}))


def train(config, args, device):
    is_meld = config['dataset'] == 'MELD'
    settings = config['stage2']
    originals, augmented = training_rows(config, args.fold)
    metadata = cache_metadata(config, args.fold, args.teacher)
    targets = load_targets(args.targets, metadata, originals)
    model, model_config = load_checkpoint(args.teacher, config, args.fold, device)
    audio_dir = config['audio_dirs']['train'] if is_meld else config['audio_dir']
    views = Views(originals, model_config, config['label_order'], aug_rows=augmented,
                  audio_dir=audio_dir, augmented_audio=settings['augmented_audio'],
                  targets=targets, augmented_text_column=settings['text_column'])
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = data_loader(views, config, device, generator=generator)
    selection_split = 'dev' if is_meld else 'heldout'
    selection = split_loader(config, model_config, args.fold, device, selection_split)
    optimizer = make_optimizer(model, config['dataset'], settings['base_lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    loss_config = dict(model_config, auxiliary_weights={
        'ctc': settings['lambda_ctc'], 'contrastive': settings['lambda_cl']})
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    initial, _ = evaluate(model, selection, model_config, device)
    print(f'Stage I {selection_split} WA: {initial["WA"]:.2f}')
    best_correct = -1
    for epoch in range(1, settings['epochs'] + 1):
        model.train()
        generator.manual_seed(args.seed + epoch + 1000 * (args.fold or 0))
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(model_config, device):
                outputs = model(batch, compute_auxiliary=True)
            loss = loss_terms(outputs, batch, loss_config, settings)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()

        scores, _ = evaluate(model, selection, model_config, device)
        scheduler.step(scores['WA'])
        if scores['correct'] > best_correct:
            best_correct = scores['correct']
            state = {'model': model.state_dict(), 'config': model_config, 'epoch': epoch,
                     'seed': args.seed, 'metrics': scores, 'stage2': settings,
                     'teacher_sha256': metadata['teacher_sha256']}
            torch.save(state, output / 'best.pt.tmp')
            (output / 'best.pt.tmp').replace(output / 'best.pt')
        print(f'Epoch {epoch}: {selection_split} WA {scores["WA"]:.2f}', flush=True)

    checkpoint = torch.load(output / 'best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model'], strict=True)
    report_split = 'test' if is_meld else selection_split
    loader = split_loader(config, model_config, args.fold, device, 'test') if is_meld else selection
    scores, records = evaluate(model, loader, model_config, device)
    write_results(output, scores, records, report_split)


def evaluate_meld(config, args, device):
    model, model_config = load_checkpoint(args.checkpoint, config, None, device)
    loader = split_loader(config, model_config, None, device, 'test')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    scores, records = evaluate(model, loader, model_config, device)
    write_results(output, scores, records, 'test')


def main(dataset, action):
    parser = argparse.ArgumentParser(description={
        'targets': 'Generate soft labels from a Stage I checkpoint.',
        'train': 'Train the MMSD Stage II student.',
        'evaluate': 'Score a selected MELD checkpoint on the test split.',
    }[action])
    root = Path(__file__).resolve().parents[1]
    parser.add_argument('--config', default=str(root / 'configs' / (dataset.lower() + '.json')))
    if dataset == 'IEMOCAP':
        parser.add_argument('--fold', type=int, choices=range(1, 6), required=True)
    else:
        parser.set_defaults(fold=None)
    parser.add_argument('--checkpoint' if action == 'evaluate' else '--teacher', required=True)
    if action == 'train':
        parser.add_argument('--targets', required=True)
    parser.add_argument('--output', required=True, help='NPZ target file' if action == 'targets' else 'Output directory')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    if config['dataset'] != dataset:
        parser.error(f'Expected a {dataset} configuration')
    settings = config['stage2']
    if settings['temperature'] <= 0 or settings['epochs'] < 1:
        parser.error('Stage II temperature and epochs must be positive')
    configure_runtime(args.seed, config)
    device = torch.device(args.device)
    {'targets': generate_targets, 'train': train, 'evaluate': evaluate_meld}[action](config, args, device)
