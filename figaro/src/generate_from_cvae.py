import os
import torch
from models.seq2seq import Seq2SeqModule
from models.cvae import VAE
from input_representation import remi2midi

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

OUTPUT_DIR = os.getenv('OUTPUT_DIR', None)
OUTPUT_FILE_NAME = os.getenv('OUTPUT_FILE_NAME', 'generation')
MAX_ITER = int(os.getenv('MAX_ITER', 16_000))
MAX_BARS = int(os.getenv('MAX_BARS', 32))

CHECKPOINT = os.getenv('CHECKPOINT', None)
CHECKPOINT_CVAE = os.getenv('CHECKPOINT_CVAE', None)
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 1))
VERBOSE = int(os.getenv('VERBOSE', 2))
CLASS_TO_GENERATE = int(os.getenv('CLASS_TO_GENERATE', 2))

# assert, that we have checkpoints to load
assert CHECKPOINT_CVAE is not None
assert CHECKPOINT is not None
assert OUTPUT_DIR is not None

verbose = VERBOSE
output_dir = OUTPUT_DIR
file = OUTPUT_FILE_NAME
c = torch.tensor(CLASS_TO_GENERATE, dtype=torch.int64)

def generate_from_thin_air():
    model = Seq2SeqModule.load_from_checkpoint(CHECKPOINT)

    encoding_size = [512, 128, 32, 16]
    decoding_size = [16, 32, 128, 512]

    # TODO: change to pytorch-lighning model loader
    # cVAE = VAE.load_from_chekpoint(CHECKPOINT_CVAE)
    cVAE = VAE(encoding_size, 8, decoding_size, conditional=True, num_labels=4)
    cVAE.load_state_dict(torch.load(CHECKPOINT_CVAE))

    sample = model.sample_cVAE(cVAE, c)

    xs_hat = sample['sequences'].detach().cpu()
    events_hat = [model.vocab.decode(x) for x in xs_hat]
    pms_hat = []

    for rec_hat in events_hat:
        pm_hat = remi2midi(rec_hat)
        pms_hat.append(pm_hat)

    if output_dir:
        for i, pm_hat in enumerate(pms_hat):
            if verbose:
                print(f"Saving to {output_dir}/{file}_{i}.mid")
            pm_hat.write(os.path.join(output_dir, file + '.mid'))

if __name__ == '__main__':
    generate_from_thin_air()