### Supervised fine-tuning
https://huggingface.co/docs/trl/en/sft_trainer

#### Setup
Install TRL (and other requirements) with 
```bash
pip install datasets transformers trl accelerate scipy sentencepiece protobuf wandb
```
Install Flash Attention with (this will take some time, be patient)
```bash
pip install ninja packaging
MAX_JOBS=6 pip install flash-attn --no-build-isolation --upgrade
```

#### Selective fine-tuning with Spectrum
```bash
git clone https://github.com/cognitivecomputations/spectrum.git
cd spectrum
```
or 
```bash
cd /raid/s3/opengptx/paramita/instruction_tuning/spectrum/
```
Then install Spectrum
```bash
pip install -r requirements.txt
```
Then launch the script
```bash
python spectrum.py --model-name <insert local or HF repo here> --top-percent <top % of snr ratios to target>
```
YAML files with the parameters to train will be saved in the same directory.

#### Run SFTTrainer
You can then adapt the hyperparameters in ``sft_run.sh`` to your preference, notably:
* ``spectrum_parameters`` should refer to the YAML file created with Spectrum
* ``neftune_noise_alpha`` can be set to enable [NEFTune](https://huggingface.co/papers/2310.05914)

Note: The script has issues when launched with ``accelerate launch`` (will investigate this).

### Preference optimization

#### Setup
Download and install `trl` from github. Make sure that you update your `accelerate` library in your environment, as older versions cause errors.

```bash
git clone https://github.com/huggingface/trl.git
cd trl
pip install -e .
cd ..
pip install accelerate --upgrade
```

#### Run preference optimization
You can then adapt the hyperparameters in `kto_run.sh` and `dpo_run.sh` to your preference and execute them to run preference optimization.
Note: A 7B model requires currently at least 2x 80G GPUs.
