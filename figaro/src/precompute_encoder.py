import os
import glob
import time
import torch
import random
from torch.utils.data import DataLoader

from models.vae import VqVaeModule
from models.seq2seq import Seq2SeqModule
from datasets import MidiDataset, SeqCollator
from utils import medley_iterator
from input_representation import remi2midi


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL = os.getenv('MODEL', '')

ROOT_DIR = os.getenv('ROOT_DIR', './lmd_full')
OUTPUT_DIR = os.getenv('OUTPUT_DIR', './samples')
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

    output_dir = "./data/remi+"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Saving generated files to: {output_dir}")

    vae_module = None

    midi_files = glob.glob(os.path.join("./data", '**/*.mid'), recursive=True)

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


def main():

    if OUTPUT_DIR:
        output_dir = os.path.join(OUTPUT_DIR, MODEL, "encoder_hidden")
    else:
        raise ValueError("OUTPUT_DIR must be specified.")

    print(f"Saving generated files to: {output_dir}")

    if VAE_CHECKPOINT:
        vae_module = VqVaeModule.load_from_checkpoint(VAE_CHECKPOINT)
        vae_module.cpu()
    else:
        vae_module = None

    model = Seq2SeqModule.load_from_checkpoint(CHECKPOINT)
    #model.to(device)
    print(f"Model is on {device}")
    model.freeze()
    model.eval()

    midi_files = glob.glob(os.path.join(ROOT_DIR, '**/*.mid'), recursive=True)

    if MAX_N_FILES > 0:
        midi_files = midi_files[:MAX_N_FILES]

    description_options = None
    if MODEL in ['figaro-no-inst', 'figaro-no-chord', 'figaro-no-meta']:
        description_options = model.description_options

    dataset = MidiDataset(
        midi_files,
        max_len=-1,
        description_flavor=model.description_flavor,
        description_options=description_options,
        max_bars=model.context_size,
        vae_module=vae_module,
        change_chord=CHANGE_CHORD
    )

    start_time = time.time()
    coll = SeqCollator(context_size=-1)
    dl = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=coll)

    if MAKE_MEDLEYS:
        dl = medley_iterator(dl,
                             n_pieces=N_MEDLEY_BARS,
                             n_bars=N_MEDLEY_BARS,
                             description_flavor=model.description_flavor
                             )

    for batch in dl:
        batch_size, seq_len = batch['input_ids'].shape[:2]

        z = {}
        desc_bar_ids = None
        if model.description_flavor in ['description', 'both']:
            z['description'] = batch['description']
            desc_bar_ids = batch['desc_bar_ids']
        if model.description_flavor in ['latent', 'both']:
            z['latents'] = batch['latents']

        if VERBOSE:
            print(f"Generating encoder_hidden for {batch['files']}")
        encoder_hidden = model.encode(z, desc_bar_ids=desc_bar_ids)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            for enc, file in zip(encoder_hidden, batch['files']):
                file_path = os.path.join(output_dir, file.replace(".mid", ".pt"))
                torch.save(enc, file_path)


if __name__ == '__main__':
    main()