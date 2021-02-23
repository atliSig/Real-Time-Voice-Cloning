import numpy as np
import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fastspeech2.model import FastSpeech2
from fastspeech2.loss import FastSpeech2Loss
from fastspeech2.dataset import Dataset
from fastspeech2.hparams import HyperParameters as hp
import fastspeech2.utils as utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_FastSpeech2(num):
    checkpoint_path = os.path.join(
        hp.checkpoint_path, "checkpoint_{}.pth.tar".format(num))
    model = nn.DataParallel(FastSpeech2())
    model.load_state_dict(torch.load(checkpoint_path)['model'])
    model.requires_grad = False
    model.eval()
    return model


def evaluate(model, step, comet_experiment=None, vocoder=None):
    torch.manual_seed(0)

    # Get dataset
    dataset = Dataset("val.txt", sort=False)
    loader = DataLoader(dataset, batch_size=hp.batch_size**2, shuffle=False,
                        collate_fn=dataset.collate_fn, drop_last=False, num_workers=0, )

    # Get loss function
    Loss = FastSpeech2Loss().to(device)

    # Evaluation
    d_l = []
    f_l = []
    e_l = []
    mel_l = []
    mel_p_l = []
    current_step = 0
    idx = 0
    for i, batchs in enumerate(loader):
        for j, data_of_batch in enumerate(batchs):
            # Get Data
            id_ = data_of_batch["id"]
            text = torch.from_numpy(data_of_batch["text"]).long().to(device)
            mel_target = torch.from_numpy(data_of_batch["mel_target"]).float().to(device)
            D = torch.from_numpy(data_of_batch["D"]).int().to(device)
            log_D = torch.from_numpy(data_of_batch["log_D"]).int().to(device)
            f0 = torch.from_numpy(data_of_batch["f0"]).float().to(device)
            energy = torch.from_numpy(data_of_batch["energy"]).float().to(device)
            src_len = torch.from_numpy(data_of_batch["src_len"]).long().to(device)
            mel_len = torch.from_numpy(data_of_batch["mel_len"]).long().to(device)
            max_src_len = np.max(data_of_batch["src_len"]).astype(np.int32)
            max_mel_len = np.max(data_of_batch["mel_len"]).astype(np.int32)

            with torch.no_grad():
                # Forward
                mel_output, mel_postnet_output, log_duration_output, f0_output, energy_output, src_mask, mel_mask, out_mel_len = model(
                    text, src_len, mel_target, mel_len, D, f0, energy, max_src_len, max_mel_len)

                # Cal Loss
                mel_loss, mel_postnet_loss, d_loss, f_loss, e_loss = Loss(log_duration_output, log_D, f0_output, f0,
                    energy_output, energy, mel_output, mel_postnet_output, mel_target, ~src_mask, ~mel_mask)

                d_l.append(d_loss.item())
                f_l.append(f_loss.item())
                e_l.append(e_loss.item())
                mel_l.append(mel_loss.item())
                mel_p_l.append(mel_postnet_loss.item())

                if vocoder is not None:
                    if hp.vocoder == 'melgan':
                        vocoder_infer =  utils.melgan_infer
                    elif hp.vocoder == 'melgan':
                        vocoder_infer = utils.waveglow_infer

                    # Run vocoding and plotting spectrogram only when the vocoder is defined
                    for k in range(len(mel_target)):
                        basename = id_[k]
                        gt_length = mel_len[k]
                        out_length = out_mel_len[k]

                        mel_target_torch = mel_target[k:k+1, :gt_length].transpose(1, 2).detach()
                        mel_target_ = mel_target[k, :gt_length].cpu().transpose(0, 1).detach()

                        mel_postnet_torch = mel_postnet_output[k:k+1, :out_length].transpose(1, 2).detach()
                        mel_postnet = mel_postnet_output[k, :out_length].cpu().transpose(0, 1).detach()

                        comet_experiment.log_audio(vocoder_infer(mel_target_torch, vocoder), hp.sampling_rate,
                                                   'eval_ground-truth_{}_{}.wav'.format(basename, hp.vocoder))
                        comet_experiment.log_audio(vocoder_infer(mel_postnet_torch, vocoder), hp.sampling_rate,
                                                   'eval_{}_{}.wav'.format(basename, hp.vocoder))
                        # np.save(os.path.join(hp.eval_path, 'eval_{}_mel.npy'.format(
                        #     basename)), mel_postnet.numpy())

                        f0_ = f0[k, :gt_length].detach().cpu().numpy()
                        energy_ = energy[k, :gt_length].detach().cpu().numpy()
                        f0_output_ = f0_output[k, :out_length].detach().cpu().numpy()
                        energy_output_ = energy_output[k, :out_length].detach().cpu().numpy()

                        utils.plot_data(
                            [(mel_postnet.numpy(), f0_output_, energy_output_), (mel_target_.numpy(), f0_, energy_)],
                            comet_experiment, ['Eval Synthetized Spectrogram', 'Eval Ground-Truth Spectrogram'])
                        idx += 1

            current_step += 1

    d_l = sum(d_l) / len(d_l)
    f_l = sum(f_l) / len(f_l)
    e_l = sum(e_l) / len(e_l)
    mel_l = sum(mel_l) / len(mel_l)
    mel_p_l = sum(mel_p_l) / len(mel_p_l)

    print("\nFastSpeech2 Step {},".format(step))
    print("Duration Loss: {}".format(d_l))
    print("F0 Loss: {}".format(f_l))
    print("Energy Loss: {}".format(e_l))
    print("Mel Loss: {}".format(mel_l))
    print("Mel Postnet Loss: {}".format(mel_p_l))

    return d_l, f_l, e_l, mel_l, mel_p_l


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, default=30000)
    args = parser.parse_args()

    # Get model
    model = get_FastSpeech2(args.step).to(device)
    print("Model Has Been Defined")
    num_param = utils.get_param_num(model)
    print('Number of FastSpeech2 Parameters:', num_param)

    # Load vocoder
    if hp.vocoder == 'melgan':
        vocoder = utils.get_melgan()
    elif hp.vocoder == 'waveglow':
        vocoder = utils.get_waveglow()

    # Init directories
    if not os.path.exists(hp.log_path):
        os.makedirs(hp.log_path)
    if not os.path.exists(hp.eval_path):
        os.makedirs(hp.eval_path)

    evaluate(model, args.step, vocoder)
