# MMSD

Code for **MMSD: A Multimodal Multitask Self-Distillation Framework for Audio-Text Emotion Recognition**.

## Setup

Use Python 3.8+ and install `requirements.txt` with matching PyTorch and torchaudio builds. Set the paths in `configs/iemocap.json` and `configs/meld.json` to your manifests, WAV files, and local wav2vec 2.0 and RoBERTa models.

Each original manifest needs `file,text,emotion,speaker` columns and a WAV named `<file>.wav` for each row. IEMOCAP uses `Ses01...` IDs and four classes (`ang,hap,neu,sad`, with excitement merged into happiness). MELD uses its official train, dev, and test splits and `meld_train_dia0_utt0`-style IDs.

Stage II takes one augmented audio-text pair per training utterance. Keep the source `file` and label in the augmented CSV. The config specifies the augmented text column. The data utilities include [back-translation](meld_code/stage2/data/augment_text.py) and a [GPT-SoVITS client](meld_code/stage2/data/run_aug.py).

## IEMOCAP

Run from the repository root and repeat for folds 1–5.

```bash
python iemocap_code/stage1/train.py --fold 1 --output outputs/iem1/stage1
python iemocap_code/stage2/generate_soft_labels.py --fold 1 --teacher outputs/iem1/stage1/best.pt --output outputs/iem1/targets.npz
python iemocap_code/stage2/train_soft.py --fold 1 --teacher outputs/iem1/stage1/best.pt --targets outputs/iem1/targets.npz --output outputs/iem1/stage2
```

Each fold selects its checkpoint by held-out-session WA and reports scores on that session. Average the five fold scores.

## MELD

```bash
python meld_code/stage1/train.py --output outputs/meld/stage1
python meld_code/stage2/evaluate_meld_class_accuracy.py --checkpoint outputs/meld/stage1/best.pt --output outputs/meld/stage1_test
python meld_code/stage2/generate_soft_labels.py --teacher outputs/meld/stage1/best.pt --output outputs/meld/targets.npz
python meld_code/stage2/train_soft.py --teacher outputs/meld/stage1/best.pt --targets outputs/meld/targets.npz --output outputs/meld/stage2
```

MELD selects checkpoints by development-set WA and reports scores on the test split. Both datasets use 60 Stage I epochs and 30 Stage II epochs. Stage II uses CE, teacher distillation, CTC, and audio-text contrastive losses. Use `--seed` on `train_soft.py` to repeat Stage II with the same teacher and targets.
