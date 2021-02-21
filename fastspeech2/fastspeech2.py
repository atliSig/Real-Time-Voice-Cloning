import torch
import torch.nn as nn

from transformer.Models import Encoder, Decoder
from transformer.Layers import PostNet
from modules import VarianceAdaptor
from utils import get_mask_from_lengths
from hparams import HyperParameters as hp
from voice_cloning.encoder import params_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, use_postnet=True, speaker_encoder=None):
        super(FastSpeech2, self).__init__()

        self.encoder = Encoder()
        if speaker_encoder:
            self.speaker_encoder = lambda x: speaker_encoder(x[:, :, :params_data.mel_n_channels])
        else:
            self.speaker_encoder = None
            hp.speaker_encoder_dim = 0

        self.variance_adaptor = VarianceAdaptor()
        self.decoder = Decoder()
        self.mel_linear = nn.Linear(hp.decoder_hidden + hp.speaker_encoder_dim, hp.n_mel_channels)

        self.use_postnet = use_postnet
        if self.use_postnet:
            self.postnet = PostNet()

    def forward(self, src_seq, src_len, mel_spec=None, mel_len=None, d_target=None, p_target=None, e_target=None, max_src_len=None,
                max_mel_len=None, d_control=1.0, p_control=1.0, e_control=1.0):

        src_mask = get_mask_from_lengths(src_len, max_src_len)
        mel_mask = get_mask_from_lengths(mel_len, max_mel_len) if mel_len is not None else None

        encoder_output = self.encoder(src_seq, src_mask)

        if self.speaker_encoder:
            speaker_embedding = self.speaker_encoder(mel_spec)
            speaker_embedding = speaker_embedding.unsqueeze(1).repeat(1, max_src_len, 1)
            encoder_output = torch.cat((encoder_output, speaker_embedding), 2)

        if d_target is not None:
            variance_adaptor_output, d_prediction, p_prediction, e_prediction, _, _ = \
                self.variance_adaptor(encoder_output, src_mask, mel_mask, d_target, p_target, e_target, max_mel_len,
                                      d_control, p_control, e_control)
        else:
            variance_adaptor_output, d_prediction, p_prediction, e_prediction, mel_len, mel_mask = \
                self.variance_adaptor(encoder_output, src_mask, mel_mask, d_target, p_target, e_target, max_mel_len,
                                      d_control, p_control, e_control)

        decoder_output = self.decoder(variance_adaptor_output, mel_mask)
        mel_output = self.mel_linear(decoder_output)

        if self.use_postnet:
            mel_output_postnet = self.postnet(mel_output) + mel_output
        else:
            mel_output_postnet = mel_output

        return mel_output, mel_output_postnet, d_prediction, p_prediction, e_prediction, src_mask, mel_mask, mel_len


if __name__ == "__main__":
    # Test
    model = FastSpeech2(use_postnet=False)
    print(model)
    print(sum(param.numel() for param in model.parameters()))
