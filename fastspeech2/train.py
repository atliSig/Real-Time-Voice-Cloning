from comet_ml import OfflineExperiment, Experiment

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import argparse
import os
import time
from pathlib import Path

from voice_cloning.encoder.inference import load_model as load_speaker_encoder

from fastspeech2.model import FastSpeech2
from fastspeech2.loss import FastSpeech2Loss
from fastspeech2.dataset import Dataset
from fastspeech2.optimizer import ScheduledOptim
from fastspeech2.evaluate import evaluate
from fastspeech2.hparams import HyperParameters as hp
import fastspeech2.utils as utils
from fastspeech2.audio import tools as audiotools


def main(args):
    torch.manual_seed(0)

    # Get device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Get dataset
    dataset = Dataset("train.txt")
    loader = DataLoader(dataset, batch_size=hp.batch_size**2, shuffle=True,
                        collate_fn=dataset.collate_fn, drop_last=True, num_workers=hp.num_workers)

    speaker_encoder = None
    if hp.speaker_encoder_path != "":
        speaker_encoder = load_speaker_encoder(Path(hp.speaker_encoder_path), device).to(device)
        for param in speaker_encoder.parameters():
                param.requires_grad = False
        else:
            speaker_encoder.train()

    # Define model
    fastspeech_model = FastSpeech2(speaker_encoder).to(device)
    model = nn.DataParallel(fastspeech_model).to(device)
    print("Model Has Been Defined")
    num_param = utils.get_param_num(model)
    print('Number of FastSpeech2 Parameters:', num_param)

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=hp.betas, eps=hp.eps, weight_decay=hp.weight_decay)
    scheduled_optim = ScheduledOptim(optimizer, hp.decoder_hidden, hp.n_warm_up_step, args.restore_step)
    Loss = FastSpeech2Loss().to(device)
    print("Optimizer and Loss Function Defined.")

    # Load checkpoint if exists
    checkpoint_path = os.path.join(hp.checkpoint_path)
    try:
        checkpoint = torch.load(os.path.join(
            checkpoint_path, 'checkpoint_{}.pth.tar'.format(args.restore_step)))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("\n---Model Restored at Step {}---\n".format(args.restore_step))
    except:
        print("\n---Start New Training---\n")
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

    # Load vocoder
    if hp.vocoder == 'melgan':
        vocoder = utils.get_melgan()
        vocoder_infer = utils.melgan_infer
    elif hp.vocoder == 'waveglow':
        vocoder = utils.get_waveglow()
        vocoder_infer = utils.waveglow_infer
    else:
        raise ValueError("Vocoder '%s' is not supported", hp.vocoder)

    comet_experiment = None
    use_comet = int(os.getenv("USE_COMET", default=0))
    if use_comet != 0:
        if use_comet == 1:
            offline_dir = os.path.join(hp.models_path, "comet")
            os.makedirs(offline_dir, exist_ok=True)
            comet_experiment = OfflineExperiment(
                project_name="mlp-project",
                workspace="ino-voice",
                offline_directory=offline_dir,
            )
        elif use_comet == 2:
            comet_experiment = Experiment(
                api_key="BtyTwUoagGMh3uN4VZt6gMOn8",
                project_name="mlp-project",
                workspace="ino-voice",
            )

        comet_experiment.set_name(args.experiment_name)
        comet_experiment.log_parameters(hp)
        comet_experiment.log_html(args.m)

    start_time = time.perf_counter()
    first_mel_train_loss, first_postnet_train_loss, first_d_train_loss, first_f_train_loss, first_e_train_loss = \
        None, None, None, None, None
    
    for epoch in range(hp.epochs):
        total_step = hp.epochs * len(loader) * hp.batch_size
        for i, batchs in enumerate(loader):
            for j, data_of_batch in enumerate(batchs):
                model = model.train()

                current_step = i * hp.batch_size + j + args.restore_step + epoch * len(loader) * hp.batch_size + 1

                # Get Data
                text = torch.from_numpy(data_of_batch["text"]).long().to(device)
                mel_target = torch.from_numpy(data_of_batch["mel_target"]).float().to(device)
                D = torch.from_numpy(data_of_batch["D"]).long().to(device)
                log_D = torch.from_numpy(data_of_batch["log_D"]).float().to(device)
                f0 = torch.from_numpy(data_of_batch["f0"]).float().to(device)
                energy = torch.from_numpy(data_of_batch["energy"]).float().to(device)
                src_len = torch.from_numpy(data_of_batch["src_len"]).long().to(device)
                mel_len = torch.from_numpy(data_of_batch["mel_len"]).long().to(device)
                max_src_len = np.max(data_of_batch["src_len"]).astype(np.int32)
                max_mel_len = np.max(data_of_batch["mel_len"]).astype(np.int32)

                # text = torch.from_numpy(data_of_batch["text"]).long()
                # mel_target = torch.from_numpy(data_of_batch["mel_target"]).float()
                # D = torch.from_numpy(data_of_batch["D"]).long()
                # log_D = torch.from_numpy(data_of_batch["log_D"]).float()
                # f0 = torch.from_numpy(data_of_batch["f0"]).float()
                # energy = torch.from_numpy(data_of_batch["energy"]).float()
                # src_len = torch.from_numpy(data_of_batch["src_len"]).long()
                # mel_len = torch.from_numpy(data_of_batch["mel_len"]).long()
                # max_src_len = np.max(data_of_batch["src_len"]).astype(np.int32)
                # max_mel_len = np.max(data_of_batch["mel_len"]).astype(np.int32)

                # Forward
                mel_output, mel_postnet_output, log_duration_output, f0_output, energy_output, src_mask, mel_mask, _ = \
                    model(text, src_len, mel_target, mel_len, D, f0, energy, max_src_len, max_mel_len)

                # Cal Loss
                mel_loss, mel_postnet_loss, d_loss, f_loss, e_loss = Loss(
                    log_duration_output, log_D, f0_output, f0, energy_output, energy, mel_output, mel_postnet_output, mel_target, ~src_mask, ~mel_mask)
                total_loss = mel_loss + mel_postnet_loss + d_loss + f_loss + e_loss

                # Set initial values for scaling
                if first_mel_train_loss is None:
                    first_mel_train_loss = mel_loss
                    first_postnet_train_loss = mel_postnet_loss
                    first_d_train_loss = d_loss
                    first_f_train_loss = f_loss
                    first_e_train_loss = e_loss

                mel_l = mel_loss.item() / first_mel_train_loss
                mel_postnet_l = mel_postnet_loss.item() / first_postnet_train_loss
                d_l = d_loss.item() / first_d_train_loss
                f_l = f_loss.item() / first_f_train_loss
                e_l = e_loss.item() / first_e_train_loss

                # Logger
                if comet_experiment is not None:
                    comet_experiment.log_metric("total_loss", mel_l+mel_postnet_l+d_l+f_l+e_l, current_step)
                    comet_experiment.log_metric("mel_loss", mel_l, current_step)
                    comet_experiment.log_metric("mel_postnet_loss", mel_postnet_l, current_step)
                    comet_experiment.log_metric("duration_loss", d_l, current_step)
                    comet_experiment.log_metric("f0_loss", f_l, current_step)
                    comet_experiment.log_metric("energy_loss", e_l, current_step)

                # Backward
                total_loss = total_loss / hp.acc_steps
                total_loss.backward()
                if current_step % hp.acc_steps != 0:
                    continue

                # Clipping gradients to avoid gradient explosion
                nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip_thresh)

                # Update weights
                scheduled_optim.step_and_update_lr()
                scheduled_optim.zero_grad()

                # Print
                if current_step % hp.log_step == 0:
                    now = time.perf_counter()

                    print("\nEpoch [{}/{}], Step [{}/{}]:".format(epoch+1, hp.epochs, current_step, total_step))
                    print("Total Loss: {:.4f}, Mel Loss: {:.5f}, Mel PostNet Loss: {:.5f}, Duration Loss: {:.5f}, "
                           "F0 Loss: {:.5f}, Energy Loss: {:.5f};".format(mel_l+mel_postnet_l+d_l+f_l+e_l, mel_l,
                                                                          mel_postnet_l, d_l, f_l, e_l))
                    print("Time Used: {:.3f}s".format(now - start_time))
                    start_time = now

                if current_step % hp.checkpoint == 0:
                    file_path = os.path.join(checkpoint_path, 'checkpoint_{}.pth.tar'.format(current_step))
                    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, file_path)
                    print("saving model at to {}".format(file_path))

                if current_step % hp.synth_step == 0:
                    length = mel_len[0].item()
                    mel_target_torch = mel_target[0, :length].detach().unsqueeze(0).transpose(1, 2)
                    mel_target = mel_target[0, :length].detach().cpu().transpose(0, 1)
                    mel_torch = mel_output[0, :length].detach().unsqueeze(0).transpose(1, 2)
                    mel = mel_output[0, :length].detach().cpu().transpose(0, 1)
                    mel_postnet_torch = mel_postnet_output[0, :length].detach().unsqueeze(0).transpose(1, 2)
                    mel_postnet = mel_postnet_output[0, :length].detach().cpu().transpose(0, 1)

                    if comet_experiment is not None:
                        comet_experiment.log_audio(audiotools.inv_mel_spec(mel),
                                                   hp.sampling_rate, "step_{}_griffin_lim.wav".format(current_step))
                        comet_experiment.log_audio(audiotools.inv_mel_spec(mel_postnet),
                                                   hp.sampling_rate, "step_{}_postnet_griffin_lim.wav".format(current_step))
                        comet_experiment.log_audio(vocoder_infer(mel_torch, vocoder), hp.sampling_rate,
                                                   'step_{}_{}.wav'.format(current_step, hp.vocoder))
                        comet_experiment.log_audio(vocoder_infer(mel_postnet_torch, vocoder), hp.sampling_rate,
                                                   'step_{}_postnet_{}.wav'.format(current_step, hp.vocoder))
                        comet_experiment.log_audio(vocoder_infer(mel_target_torch, vocoder), hp.sampling_rate,
                                                   'step_{}_ground-truth_{}.wav'.format(current_step, hp.vocoder))

                        f0 = f0[0, :length].detach().cpu().numpy()
                        energy = energy[0, :length].detach().cpu().numpy()
                        f0_output = f0_output[0, :length].detach().cpu().numpy()
                        energy_output = energy_output[0, :length].detach().cpu().numpy()

                        utils.plot_data([(mel_postnet.numpy(), f0_output, energy_output), (mel_target.numpy(), f0, energy)],
                            comet_experiment, ['Synthesized Spectrogram', 'Ground-Truth Spectrogram'])

                if current_step % hp.eval_step == 0:
                    model.eval()
                    with torch.no_grad():
                        if comet_experiment is not None:
                            with comet_experiment.validate():
                                d_l, f_l, e_l, m_l, m_p_l = evaluate(model, current_step, comet_experiment)
                                t_l = d_l + f_l + e_l + m_l + m_p_l

                                comet_experiment.log_metric("total_loss", t_l, current_step)
                                comet_experiment.log_metric("mel_loss", m_l, current_step)
                                comet_experiment.log_metric("mel_postnet_loss", m_p_l, current_step)
                                comet_experiment.log_metric("duration_loss", d_l, current_step)
                                comet_experiment.log_metric("F0_loss", f_l, current_step)
                                comet_experiment.log_metric("energy_loss", e_l, current_step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_step', type=int, default=0)
    parser.add_argument('--experiment-name', type=str, required=True)
    parser.add_argument('-m', type=str, default="")
    args = parser.parse_args()
    main(args)
