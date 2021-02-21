import sys
import os
from pathlib import Path

from voice_cloning.encoder.params_model import model_embedding_size as speaker_embedding_size
from voice_cloning.utils.argutils import print_args
from voice_cloning.encoder import inference as encoder

import numpy as np
import soundfile as sf
import librosa
import argparse
import torch
from audioread.exceptions import NoBackendError

if __name__ == '__main__':
    ## Info & args
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-e", "--enc_model_fpath", type=Path,
        default="./voice_cloning/encoder/saved_models/pretrained.pt",
        help="Path to a saved encoder")
    parser.add_argument("--audio_dir",
        default='./speaker_verification/data/ground_truth_audio',
        help="A path to a directory of ground truth audio")
    parser.add_argument("--output_dir",
        default='./speaker_verification/data/embeddings',
        help="A path to a directory where generated embeddings will be stored")
    args = parser.parse_args()
    print_args(args, parser)

    encoder.load_model(args.enc_model_fpath)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    for subdir, dirs, files in os.walk(args.audio_dir):
        for audio_file in files:
            audio_fid, ext = os.path.splitext(audio_file)
            if ext in ['.wav', '.mp3']:
                in_fpath = Path(os.path.join(subdir, audio_file))
                out_fpath = Path(os.path.join(
                    args.output_dir,
                    *in_fpath.parent.parts[-2:],
                    f'{audio_fid}.npy'))
                Path(out_fpath.parent).mkdir(parents=True, exist_ok=True)

                preprocessed_wav = encoder.preprocess_wav(in_fpath)
                original_wav, sampling_rate = librosa.load(str(in_fpath))
                preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)

                embed = encoder.embed_utterance(preprocessed_wav)
                np.save(out_fpath, embed)