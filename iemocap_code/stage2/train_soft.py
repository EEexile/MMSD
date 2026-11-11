"""MMSD student training."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mmsd.stage2 import main


if __name__ == '__main__':
    main('IEMOCAP', 'train')
