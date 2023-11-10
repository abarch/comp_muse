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
cd ..
```

```bash
conda activate figaro
cd EMOPIA_cls
pip install -r requirements.txt
cd ..
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

You can choose which FIGARO model you want to use by changing CHECKPOINT and MODEL. The options are: figaro-expert, figaro-learned and figaro (for both)
```bash
cd figaro
ROOT_DIR={where the data is saved} CHECKPOINT=./checkpoints/figaro.ckpt MODEL=figaro VAE_CHECKPOINT=./checkpoints/vq-vae.ckpt python src/precompute_encoder.py
```
3. Train the CVAE
```bash
MODEL=figaro python src/trainCVAE.py
```
## Training EMOPIA classifier
1. Precompute the Remi+ dataset
```bash
cd EMOPIA_cls
ROOT_DIR={where the data is saved} python ../figaro/src/precompute_remi_plus.py   
```
2. Train the EMOPIA classifier

By changing `--task` you can choose which quadrants are learned. 
The options are `ar_va` for all quadrants,
`h_l` for Q1 (high arousal & high valence) and Q3 (low arousal & low valence),
`arousal` (high arousal = Q1 & Q2; low arousal = Q3 & Q4)
and `valence` (high valence = Q1 & Q4; low valence = Q2 & Q3).
```bash
cd midi_cls
python train_test.py --midi remi+ --task {ar_va or h_l} --gpus 0 --save_best_weight True --batch_size 32
```
## Generating samples
Options to change:
* `CLASSIFY`: Whether to classify the generated song by the trained classifier
* `CLS_TASK`: Which classifier to use.
* `CLASS_TO_GENERATE`: The class to generate. Set it to -1 to generate songs per class. 
* `RESAMPLE_PER_BAR`: Whether to generate a new encoder hidden state every bar or only at the beginning of the song.
* `CHECKPOINT`: Path to the FIGARO checkpoint. You can also use the figaro-learned or figaro-expert model.
* `CHECKPOINT_CVAE`: Path to the CVAE checkpoint. Depends on which FIGARO model you used to generate the training data.
* `OUTPUT_DIR`: Where the music is saved.
- Generate samples from CVAE without classification
```bash
cd figaro
CLASSIFY=False CHECKPOINT=./checkpoints/figaro.ckpt CHECKPOINT_CVAE=./training_saves/figaro_vae_hidden_model.ckpt CLASS_TO_GENERATE=-1 OUTPUT_DIR=./samples/figaro/cvae_generated RESAMPLE_PER_BAR=True python src/generate_from_cvae.py
```
- Generate samples from CVAE with classification
```bash
cd figaro
CLASSIFY=True CLS_TASK={ar_va or h_l} CLS_TYPES=remi+ CHECKPOINT=./checkpoints/figaro.ckpt CHECKPOINT_CVAE=./training_saves/figaro_vae_hidden_model.ckpt CLASS_TO_GENERATE=-1 OUTPUT_DIR=./samples/figaro/cvae_generated RESAMPLE_PER_BAR=True python src/generate_from_cvae.py
```
