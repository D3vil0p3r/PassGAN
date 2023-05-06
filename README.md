# PassGAN

This repository contains code for the [_PassGAN: A Deep Learning Approach for Password Guessing_](https://arxiv.org/abs/1709.00440) paper. 

The model from PassGAN is taken from [_Improved Training of Wasserstein GANs_](https://arxiv.org/abs/1704.00028) and it is assumed that the authors of PassGAN used the [improved_wgan_training](https://github.com/igul222/improved_wgan_training) tensorflow implementation in their work. For this reason, I have modified that reference implementation in this repository to make it easy to train (`passgan.py train`) and sample (`passgan.py sample`) from. This repo contributes:

- A command-line interface
- A pretrained PassGAN model trained on the RockYou dataset

```
          _______           __________           ___
    ____            _____                __
__      ____                  _________    _   __
       / __ \____ ___________/ ____/   |  / | / /
   _  / /_/ / __ `/ ___/ ___/ / __/ /| | /  |/ / 
___  / ____/ /_/ (__  |__  ) /_/ / ___ |/ /|  /  
    /_/    \__,_/____/____/\____/_/  |_/_/ |_/   
  __        _______         ____            __
        ____           __________    _____

A Deep Learning Approach for Password Guessing.

List of arguments:

-h, --help              show this help message and exit
sample                  use the pretrained model to generate passwords
train                   train a model on a large dataset (can take several hours on a GTX 1080)

Usage Examples:
passgan sample --input-dir pretrained --checkpoint pretrained/checkpoints/checkpoint_5000.ckpt --output gen_passwords.txt --batch-size 1024 --num-samples 1000000
passgan train --output-dir pretrained --training-data data/train.txt
```

## Getting Started

### Arch-based distributions
```bash
# requires CUDA 8 to be pre-installed
pacman -S python-matplotlib python-numpy python-tensorflow
```

### Training your own models

Training a model on a large dataset (100MB+) can take several hours on a GTX 1080.

If you don't want to wait, jump to [Generating password samples](https://github.com/D3vil0p3r/PassGAN/tree/main#generating-password-samples) section and use the `pretrained` folder in this repository as `--input-dir`.

```bash
# download the rockyou training data
# contains 80% of the full rockyou passwords (with repeats)
# that are 10 characters or less
curl -L -o data/train.txt https://github.com/brannondorsey/PassGAN/releases/download/data/rockyou-train.txt

# train for 200000 iterations, saving checkpoints every 5000
# uses the default hyperparameters from the paper
python passgan.py train --output-dir pretrained --training-data data/train.txt
```

You are encouraged to train using your own password leaks and datasets. Some great places to find those include:

- [LinkedIn leak](https://github.com/brannondorsey/PassGAN/releases/download/data/68_linkedin_found_hash_plain.txt.zip) (1.7GB compressed, direct download. Mirror from [Hashes.org](https://hashes.org/leaks.php))
- [Exploit.in torrent](https://thepiratebay.org/torrent/16016494/exploit.in) (10GB+, 800 million accounts. Infamous!)
- [Hashes.org](https://hashes.org/leaks.php): Awesome shared password recovery site. Consider donating if you have the resources ;)

### Generating password samples

Use the pretrained model to generate 1,000,000 passwords, saving them to `gen_passwords.txt`.

```bash
python passgan.py sample \
	--input-dir pretrained \
	--checkpoint pretrained/checkpoints/checkpoint_5000.ckpt \
	--output gen_passwords.txt \
	--batch-size 1024 \
	--num-samples 1000000
```

## Results

I've yet to do an exhaustive analysis of my attempt to reproduce the results from the PassGAN paper. However, using the pretrained rockyou model to generate 10⁸ password samples I was able to match 630,347 (23.97%) unique passwords in the test data, using a 80%/20% train/test split.

In general, I am somewhat surprised (and dissapointed) that the authors of PassGAN referenced [prior work](https://www.usenix.org/system/files/conference/usenixsecurity16/sec16_paper_melicher.pdf) in the ML password generation domain but did not compare their results to that research. My initial experience with PassGAN leads me to believe that it would significantly underperform both the RNN and Markov-based approaches mentioned in that paper and I hope that it is not for this reason that the authors have chosen not to compare results.

## Attribution and License

This code is released under an [MIT License](https://github.com/igul222/improved_wgan_training/blob/master/LICENSE). You are free to use, modify, distribute, or sell it under those terms. 

The majority of the credit for the code in this repository goes to @igul222 for his work on the [improved_wgan_training](https://github.com/igul222/improved_wgan_training). I've simply modularized his code a bit, added a command-line interface, and specialized it for the PassGAN paper.

The PassGAN [research and paper](https://arxiv.org/abs/1709.00440) was published by Briland Hitaj, Paolo Gasti, Giuseppe Ateniese, Fernando Perez-Cruz.
