import os
import glob
import time
import torch
from torch.utils.data import DataLoader

from models.vae import VqVaeModule
from models.seq2seq import Seq2SeqModule
from datasets import MidiDataset, SeqCollator
from utils import medley_iterator


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL = os.getenv('MODEL', '')

ROOT_DIR = os.getenv('ROOT_DIR', '../../dataset')
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

        offset = min(256, z['description'].size(1))
        desc_bar_ids_ = torch.zeros(batch_size, 256, dtype=torch.int)
        z_ = torch.zeros(batch_size, 256, dtype=torch.int)
        z_[0,:offset] = z['description'][0, :offset]
        z['description'] = z_
        desc_bar_ids_[0,:offset] = desc_bar_ids[0, :offset]
        encoder_hidden = model.encode(z, desc_bar_ids=desc_bar_ids_)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            for enc, file in zip(encoder_hidden, batch['files']):
                new_file = file.replace(".mid", ".pt")
                new_file = os.path.join(new_file[0:2], new_file[3:])
                file_path = os.path.join(output_dir, new_file)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                torch.save(enc, file_path)


if __name__ == '__main__':
    main()