import os
import glob
import torch
from torch.utils.data import DataLoader
from datasets import MidiDataset, SeqCollator


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL = os.getenv('MODEL', '')

ROOT_DIR = os.getenv('ROOT_DIR', '../../dataset')
OUTPUT_DIR = os.getenv('OUTPUT_DIR', './dataset/remi+')
MAX_N_FILES = int(float(os.getenv('MAX_N_FILES', -1)))
MAX_ITER = int(os.getenv('MAX_ITER', 16_000))
MAX_BARS = int(os.getenv('MAX_BARS', 32))

MAKE_MEDLEYS = os.getenv('MAKE_MEDLEYS', 'False') == 'True'
N_MEDLEY_PIECES = int(os.getenv('N_MEDLEY_PIECES', 2))
N_MEDLEY_BARS = int(os.getenv('N_MEDLEY_BARS', 16))

CHECKPOINT = os.getenv('CHECKPOINT', None)
VAE_CHECKPOINT = os.getenv('VAE_CHECKPOINT', None)
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 1))
VERBOSE = int(os.getenv('VERBOSE', 2))

CHANGE_CHORD = os.getenv('CHANGE_CHORD', 'None')  # Possible options: "to_other", "to_min", "to_maj"


def midi_to_remi_plus():

    output_dir = OUTPUT_DIR
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Saving generated files to: {output_dir}")

    vae_module = None

    midi_files = glob.glob(os.path.join(ROOT_DIR, '**/*.mid'), recursive=True)

    description_options = None

    dataset = MidiDataset(
        midi_files,
        max_len=-1,
        description_options=description_options,
        max_bars=2048,
        max_positions=2048,
        vae_module=vae_module
    )

    coll = SeqCollator(context_size=-1)
    dl = DataLoader(dataset, batch_size=1, collate_fn=coll)


    with torch.no_grad():
        for batch in dl:
            remi_p = batch["input_ids"].squeeze().tolist()
            remi_p_fn = os.path.join(output_dir, batch["files"][0].replace(".mid",".pt"))
            torch.save(remi_p, remi_p_fn)


if __name__ == '__main__':
    midi_to_remi_plus()