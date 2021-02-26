import os
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class HyperParameters:
    experiment_name = "experiment_name"
    # Dataset
    num_workers: int = 4
    dataset: str = "VCTK"
    data_path: str = os.path.join(os.environ.get('DATASET_DIR', "/home/rokas/year4/mlp/cw3/data/datasets/"), 'VCTK')
    models_path: str = os.environ.get('MODEL_DIR', "/home/rokas/year4/mlp/cw3/data/models/fastspeech2")

    # Text
    text_cleaners: List = field(default_factory=lambda: ['english_cleaners'])

    sampling_rate: int = 22050
    filter_length: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mel_channels: int = 80
    mel_fmin: float = 0.0
    mel_fmax: float = 8000.0

    # FastSpeech 2
    encoder_layer: int = 4
    encoder_head: int = 2
    encoder_hidden: int = 256
    decoder_layer: int = 4
    decoder_head: int = 2
    decoder_hidden: int = 256
    fft_conv1d_filter_size: int = 1024
    fft_conv1d_kernel_size: Tuple = (9, 1)
    encoder_dropout: float = 0.2
    decoder_dropout: float = 0.2

    variance_predictor_filter_size: int = 256
    variance_predictor_kernel_size: int = 3
    variance_predictor_dropout: float = 0.5

    max_seq_len: int = 1000

    # Speaker Encoder
    speaker_encoder_dim: int = 256
    speaker_encoder_path: str = "/home/rokas/year4/mlp/cw3/data/models/speaker_encoder/pretrained.pt"
    #speaker_encoder_path: str = ""
    train_speaker_encoder: bool = False

    ### LJSpeech ###
    # f0_min: float = 71.0
    # f0_max: float = 795.8
    # energy_min: float = 0.0
    # energy_max: float = 315.0
    # n_bins: int = 256
    # max_wav_value: float = 32768.0

    ### VCTK ###
    f0_min: float = 71.0
    f0_max: float = 797.9
    energy_min: float = 0.0
    energy_max: float = 447.0
    n_bins: int = 256
    max_wav_value: float = 38210.0

    # Checkpoints and synthesis path
    preprocessed_path: str = os.path.join(data_path, "preprocessed")
    checkpoint_path: str = os.path.join(models_path, "ckpt")
    synth_path: str = os.path.join(models_path, "synth")
    eval_path: str = os.path.join(models_path, "eval")
    log_path: str = os.path.join(models_path, "log")
    test_path: str = os.path.join(models_path, "results")

    # Optimizer
    lr: float = 1e-4
    batch_size: int = 4
    epochs: int = 1
    n_warm_up_step: int = 4000
    grad_clip_thresh: float = 1.0
    acc_steps: int = 1
    betas: Tuple = (0.9, 0.98)
    eps: float = 1e-9
    weight_decay: float = 0.

    # Vocoder
    vocoder: str = 'waveglow' #'waveglow' or 'melgan'

    # Log-scaled duration
    log_offset: float = 1.

    # Save, log and synthesis
    checkpoint: int = 1000
    synth_step: int = 1000
    eval_step: int = 1000
    log_step: int = 1000

