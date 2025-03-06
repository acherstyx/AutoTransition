# Won et al. 2020
# Data-driven harmonic filters for audio representation learning
# https://github.com/minzwon/data-driven-harmonic-filters
# https://github.com/minzwon/sota-music-tagging-models

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


# Multi-GPU --> change batch norm to weight norm

def hz_to_midi(hz):
    return 12 * (torch.log2(hz) - np.log2(440.0)) + 69


def midi_to_hz(midi):
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


def note_to_hz(note):
    return librosa.core.note_to_hz(note)


def note_to_midi(note):
    return librosa.core.note_to_midi(note)


def hz_to_note(hz):
    return librosa.core.hz_to_note(hz)


def initialize_filterbank(sample_rate, n_harmonic, semitone_scale):
    # MIDI
    # lowest note
    low_midi = note_to_midi('C1')

    # highest note
    high_note = hz_to_note(sample_rate / (2 * n_harmonic))
    high_midi = note_to_midi(high_note)

    # number of scales
    level = (high_midi - low_midi) * semitone_scale
    midi = np.linspace(low_midi, high_midi, level + 1)
    hz = midi_to_hz(midi[:-1])

    # stack harmonics
    harmonic_hz = []
    for i in range(n_harmonic):
        harmonic_hz = np.concatenate((harmonic_hz, hz * (i + 1)))

    return harmonic_hz, level


def l2_norm(x):
    if len(x.shape):
        x = x.reshape((x.shape[0], -1))
    return F.normalize(x, p=2, dim=1)


class metric_mlp(torch.nn.Module):

    def __init__(self, input_dim=256, output_dim=100):
        super().__init__()

        hidden_dim0 = 100
        self.Linear0 = nn.utils.weight_norm(nn.Linear(input_dim, hidden_dim0))
        self.Linear1 = nn.Linear(hidden_dim0, output_dim)

        self.leaky_relu = nn.LeakyReLU()
        self.bn0 = nn.BatchNorm1d(hidden_dim0)

    def forward(self, x):
        x = self.Linear0(x)
        x = self.leaky_relu(x)
        x = self.bn0(x)
        x = self.Linear1(x)
        return x


class HarmonicSTFT(nn.Module):
    def __init__(self,
                 sample_rate=16000,
                 n_fft=513,
                 win_length=None,
                 hop_length=None,
                 pad=0,
                 power=2,
                 normalized=False,
                 n_harmonic=6,
                 semitone_scale=2,
                 bw_Q=1.0,
                 learn_bw=None):
        super(HarmonicSTFT, self).__init__()

        # Parameters
        self.sample_rate = sample_rate
        self.n_harmonic = n_harmonic
        self.bw_alpha = 0.1079
        self.bw_beta = 24.7

        # Spectrogram
        self.spec = torchaudio.transforms.Spectrogram(n_fft=n_fft, win_length=win_length,
                                                      hop_length=hop_length, pad=0,
                                                      window_fn=torch.hann_window,
                                                      power=power, normalized=normalized, wkwargs=None)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

        # Initialize the filterbank. Equally spaced in MIDI scale.
        harmonic_hz, self.level = initialize_filterbank(sample_rate, n_harmonic, semitone_scale)

        # Center frequncies to tensor
        self.f0 = torch.tensor(harmonic_hz.astype('float32'))

        # Bandwidth parameters
        if learn_bw == 'only_Q':
            self.bw_Q = nn.Parameter(torch.tensor(np.array([bw_Q]).astype('float32')))
        elif learn_bw == 'fix':
            self.bw_Q = torch.tensor(np.array([bw_Q]).astype('float32'))

    def get_harmonic_fb(self):
        # bandwidth
        bw = (self.bw_alpha * self.f0 + self.bw_beta) / self.bw_Q
        bw = bw.unsqueeze(0)  # (1, n_band)
        f0 = self.f0.unsqueeze(0)  # (1, n_band)
        fft_bins = self.fft_bins.unsqueeze(1)  # (n_bins, 1)

        up_slope = torch.matmul(fft_bins, (2 / bw)) + 1 - (2 * f0 / bw)
        down_slope = torch.matmul(fft_bins, (-2 / bw)) + 1 + (2 * f0 / bw)
        fb = torch.max(self.zero, torch.min(down_slope, up_slope))
        return fb

    def to_device(self, device, n_bins):
        self.f0 = self.f0.to(device)
        self.bw_Q = self.bw_Q.to(device)
        # fft bins
        self.fft_bins = torch.linspace(0, self.sample_rate // 2, n_bins)
        self.fft_bins = self.fft_bins.to(device)
        self.zero = torch.zeros(1)
        self.zero = self.zero.to(device)

    def forward(self, waveform):
        # stft
        spectrogram = self.spec(waveform)

        # to device
        self.to_device(waveform.device, spectrogram.size(1))

        # triangle filter
        harmonic_fb = self.get_harmonic_fb()
        harmonic_spec = torch.matmul(spectrogram.transpose(1, 2), harmonic_fb).transpose(1, 2)

        # (batch, channel, length) -> (batch, harmonic, f0, length)
        b, c, l = harmonic_spec.size()
        harmonic_spec = harmonic_spec.view(b, self.n_harmonic, self.level, l)

        # amplitude to db
        harmonic_spec = self.amplitude_to_db(harmonic_spec)
        return harmonic_spec


class Conv_2d(nn.Module):
    def __init__(self, input_channels, output_channels, shape=3, stride=1, pooling=2):
        super(Conv_2d, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, shape, stride=stride, padding=shape // 2)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(pooling)

    def forward(self, x):
        out = self.bn(self.relu(self.conv(x)))
        out = self.mp(out)
        return out


class Res_2d_mp(nn.Module):
    def __init__(self, input_channels, output_channels, pooling=2):
        super(Res_2d_mp, self).__init__()
        self.conv_1 = nn.Conv2d(input_channels, output_channels, 3, padding=1)
        self.bn_1 = nn.BatchNorm2d(output_channels)
        self.conv_2 = nn.Conv2d(output_channels, output_channels, 3, padding=1)
        self.bn_2 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(pooling)

    def forward(self, x):
        #         out = self.conv_2(self.relu(self.conv_1(x)))
        #         out = x + out
        #         out = self.mp(self.relu(out))

        out = self.conv_1(self.bn_1(self.relu(x)))
        out = self.conv_2(self.bn_2(self.relu(out)))
        out = x + out
        # relu --> mp == mp --> relu
        out = self.mp(out)

        return out


class HarmonicCNN(nn.Module):
    '''
    Won et al. 2020
    Data-driven harmonic filters for audio representation learning.
    Trainable harmonic band-pass filters, short-chunk CNN.
    '''

    def __init__(self,
                 n_channels=128,
                 sample_rate=16000,
                 n_fft=512,
                 f_min=0.0,
                 f_max=8000.0,
                 n_mels=128,
                 n_class=50,
                 n_harmonic=6,
                 semitone_scale=2,
                 learn_bw='only_Q',
                 output_embed=True):
        super(HarmonicCNN, self).__init__()

        self.output_embed = output_embed
        # Harmonic STFT
        self.hstft = HarmonicSTFT(sample_rate=sample_rate,
                                  n_fft=n_fft,
                                  n_harmonic=n_harmonic,
                                  semitone_scale=semitone_scale,
                                  learn_bw=learn_bw)
        self.hstft_bn = nn.BatchNorm2d(n_harmonic)

        # CNN
        self.layer1 = Conv_2d(n_harmonic, n_channels, pooling=2)
        self.layer2 = Res_2d_mp(n_channels, n_channels, pooling=2)
        self.layer3 = Res_2d_mp(n_channels, n_channels, pooling=2)
        self.layer4 = Res_2d_mp(n_channels, n_channels, pooling=2)
        self.layer5 = Conv_2d(n_channels, n_channels * 2, pooling=2)
        self.layer6 = Res_2d_mp(n_channels * 2, n_channels * 2, pooling=(2, 3))
        self.layer7 = Res_2d_mp(n_channels * 2, n_channels * 2, pooling=(2, 3))

        # Dense
        self.dense1 = nn.Linear(n_channels * 2, n_channels * 2)
        self.bn = nn.BatchNorm1d(n_channels * 2)
        self.dense2 = nn.Linear(n_channels * 2, n_class)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Spectrogram
        # Not sure if change to weight norm work
        x = self.hstft_bn(self.hstft(x))
        # x = nn.utils.weight_norm(self.hstft(x))

        # CNN
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = x.squeeze(2)

        # Global Max Pooling
        if x.size(-1) != 1:
            x = nn.MaxPool1d(x.size(-1))(x)
        x = x.squeeze(2)

        # Dense
        x = self.dense1(x)
        x = self.relu(x)
        x = self.bn(x)

        if self.output_embed:
            return x

        x = self.dropout(x)
        x = self.dense2(x)
        x = nn.Sigmoid()(x)
        return x
