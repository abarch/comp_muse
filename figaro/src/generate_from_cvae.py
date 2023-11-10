import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from models.seq2seq import Seq2SeqModule
from models.cvae import VAE
from input_representation import remi2midi
from EMOPIA_cls.midi_cls.src.model.net import SAN



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

CLASSIFY = os.getenv('CLASSIFY', 'False') == 'True'
CLS_TYPES = os.getenv('CLS_TYPES', 'remi+')
CLS_TASK = os.getenv('CLS_TASK', 'h_l')

RESAMPLE_PER_BAR = os.getenv('RESAMPLE_PER_BAR', 'False') == 'True'


# assert, that we have checkpoints to load
assert CHECKPOINT_CVAE is not None
assert CHECKPOINT is not None
assert OUTPUT_DIR is not None

verbose = VERBOSE
output_dir = OUTPUT_DIR
file = OUTPUT_FILE_NAME
class_to_generate = torch.tensor(CLASS_TO_GENERATE, dtype=torch.int64) - 1

def generate_from_thin_air():
    model = Seq2SeqModule.load_from_checkpoint(CHECKPOINT).to(device)

    encoding_size = [131072, 1024, 128, 32, 16]
    decoding_size = [16, 32, 128, 1024, 131072]

    # TODO: change to pytorch-lightning model loader
    # cVAE = VAE.load_from_checkpoint(CHECKPOINT_CVAE)
    cVAE = VAE(encoding_size, 8, decoding_size, conditional=True, num_labels=4)
    cVAE.load_state_dict(torch.load(CHECKPOINT_CVAE))
    if class_to_generate >= 0 and class_to_generate <= 3:
        classes = [class_to_generate]
    elif CLS_TASK == "h_l":
        classes = [2] * 5 + [0] * 5
    else:
        classes = [0] * 5 + [1] * 5 + [2] * 5 + [3] * 5

    classes = torch.tensor(classes, dtype=torch.int64)
    correct = [False]*len(classes)
    for num, c in enumerate(classes):
        sample = model.sample_cVAE(cVAE, c, max_length=2000, resample_per_bar=RESAMPLE_PER_BAR)

        xs_hat = sample['sequences'].detach().cpu()

        if CLASSIFY:
            config_path = Path("..", "EMOPIA_cls", "best_weight", CLS_TYPES, CLS_TASK, "hparams.yaml")
            checkpoint_path = Path("..", "EMOPIA_cls", "best_weight", CLS_TYPES, CLS_TASK, "best.ckpt")
            config = OmegaConf.load(config_path)
            label_list = list(config.task.labels)

            emopia_cls_model = SAN(
               num_of_dim=config.task.num_of_dim,
               vocab_size=config.midi.pad_idx + 1,
               lstm_hidden_dim=config.hparams.lstm_hidden_dim,
               embedding_size=config.hparams.embedding_size,
               r=config.hparams.r)
            state_dict = torch.load(checkpoint_path)
            new_state_map = {model_key: model_key.split("model.")[1] for model_key in state_dict.get("state_dict").keys()}
            new_state_dict = {new_state_map[key]: value for (key, value) in state_dict.get("state_dict").items() if
                              key in new_state_map.keys()}
            emopia_cls_model.load_state_dict(new_state_dict)
            emopia_cls_model.eval()
            prediction = emopia_cls_model(xs_hat)
            pred_label = label_list[prediction.squeeze(0).max(0)[1].detach().cpu().numpy()]
            pred_value = prediction.squeeze(0).detach().cpu().numpy()
            print("========")
            print(f"Emotion: Q{c+1}, classified as {pred_label}")
            print("Inference values: ", pred_value)
            if CLS_TASK == "ar_va":
                if c == np.argmax(pred_value):
                    correct[num] = True
            elif CLS_TASK == "h_l":
                if (c == 0 and np.argmax(pred_value) == 0) or (c == 2 and np.argmax(pred_value) == 1):
                    correct[num] = True

        events_hat = [model.vocab.decode(x) for x in xs_hat]
        pms_hat = []

        for rec_hat in events_hat:
            pm_hat = remi2midi(rec_hat)
            pms_hat.append(pm_hat)

        if output_dir:
            for i, pm_hat in enumerate(pms_hat):
                if verbose:
                    print(f"Saving to {file}_class={c+1}_{num}_{i}.mid")
                os.makedirs(output_dir, exist_ok=True)
                pm_hat.write(os.path.join(output_dir, f"{file}_class={c+1}_{num}_{i}.mid"))
    print(f"Correct: Accuracy={sum(correct)/len(correct)}")

if __name__ == '__main__':
    generate_from_thin_air()