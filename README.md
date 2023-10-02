# comp_muse
## Getting started
Prerequisites:
- Python 3.9
- Conda

### Setup
1. Clone this repository to your disk
3. Install required packages (see requirements.txt).
With Conda:
```bash
cd figaro
conda create --name figaro python=3.9
conda activate figaro
pip install -r requirements.txt
```

### Preparing the Data

To train models and to generate new samples, we use the [Lakh MIDI](https://colinraffel.com/projects/lmd/) dataset (altough any collection of MIDI files can be used).
1. Download (size: 1.6GB) and extract the archive file:
```bash
wget http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz
tar -xzf lmd_full.tar.gz
```
2. You may wish to remove the archive file now: `rm lmd_full.tar.gz`

### Download Pre-Trained Models
If you don't wish to train your own models, you can download our pre-trained models.
1. Download (size: 2.3GB) and extract the archive file:
```bash
wget -O checkpoints.zip https://polybox.ethz.ch/index.php/s/a0HUHzKuPPefWkW/download
unzip checkpoints.zip
```
2. You may wish to remove the archive file now: `rm checkpoints.zip`

## Training CVAE
1. Download the data files from [HERE](https://zenodo.org/record/5090631#.YQEZZ1Mzaw5).
2. Precompute the encoder hidden states
```bash
cd figaro
ROOT_DIR={where the data is saved} CHECKPOINT=./checkpoints/figaro.ckpt MODEL=figaro VAE_CHECKPOINT=./checkpoints/vq-vae.ckpt python src/precompute_encoder.py
```
3. Train the CVAE
```bash
python src/trainCVAE.py
```
## Training EMOPIA classifier
1. Precompute the Remi+ dataset
```bash
cd EMOPIA_cls
ROOT_DIR={where the data is saved} python ../figaro/src/precompute_remi_plus.py   
```
2. Train the EMOPIA classifier
```bash
cd midi_cls
python train_test.py --midi remi+ --task {ar_va or h_l} --gpus 0 --save_best_weight True --batch_size 32
```
## Generating samples
- Generate samples from CVAE without classification
```bash
cd figaro
CLASSIFY=False CHECKPOINT=./checkpoints/figaro.ckpt CHECKPOINT_CVAE=./vae_hidden_model.ckpt CLASS_TO_GENERATE=-1 OUTPUT_DIR=./samples/figaro/cvae_generated python src/generate_from_cvae.py
```
- Generate samples from CVAE with classification
```bash
cd figaro
CLASSIFY=True CLS_TASK={ar_va or h_l} CLS_TYPES=remi+ CHECKPOINT=./checkpoints/figaro.ckpt CHECKPOINT_CVAE=./vae_hidden_model.ckpt CLASS_TO_GENERATE=-1 OUTPUT_DIR=./samples/figaro/cvae_generated python src/generate_from_cvae.py
```
