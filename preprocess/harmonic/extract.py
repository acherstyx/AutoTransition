import logging
import moviepy.editor as mp
import numpy as np
import torch
import traceback
import warnings
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils import collate

from .harmonic_cnn import HarmonicCNN, metric_mlp, l2_norm

dim_audio_fea = 256  # dim
dim_metric = 100

torch.multiprocessing.set_sharing_strategy('file_system')
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


def decode_wav_to_wav_bytes(video, sample_rate=16000):
    try:
        vc = mp.VideoFileClip(video)
        audio_wave = np.array([a for a, b in vc.audio.to_soundarray(fps=sample_rate, nbytes=8)])
        return audio_wave
    except Exception as e:
        logger.warning("Failed to decode audio wave from video: %s \nFull error stack for debugging: %s",
                       video, traceback.format_exc())
        return None


class ExtractDataloaderSet(Dataset):

    def __init__(self, audio, feature_sample_rate: int, sample_rate: int, batch_size: int, args, at_time=None):
        self.audio = audio
        self.feature_sample_rate = feature_sample_rate
        self.sample_rate = sample_rate
        self.args = args
        self.batch_size = batch_size
        self.at_time = at_time

        assert self.at_time is None or len(self.audio) == len(self.at_time), "length not match"

    def __len__(self):
        return len(self.audio)

    @staticmethod
    def _calc_sampled_time(duration, sample_rate: int = 1, offset: float = 0.5, left: float = 0.5):
        """
        sample timestamp
        :param duration:
        :param sample_rate:
        """
        step = 1 / sample_rate
        i = 0
        ret = []
        while offset + i * step + left < duration:
            ret.append(offset + i * step)
            i += 1
        return ret

    def __getitem__(self, index):
        file = self.audio[index]

        audio = decode_wav_to_wav_bytes(file)
        if audio is None:
            return None, None, None
        audio_len = round(len(audio) / self.sample_rate, 2)

        if self.at_time is None:
            at_times = self._calc_sampled_time(duration=audio_len,
                                               sample_rate=self.feature_sample_rate,
                                               offset=0.5,
                                               left=0.5)
        else:
            at_times = self.at_time[index]

        intervals = []
        for t in at_times:
            if t < audio_len:
                intervals.append([(max(0, t - 1), t + 1), t, 0])
            else:
                intervals.append([(audio_len - 2, audio_len), audio_len - 1, 0])
        if not intervals:
            raise ValueError("No input timestamps are valid.")

        downbeats = AudioSamples(audio, intervals, self.args)
        batch_data = []
        data = []
        for d in downbeats:
            if len(data) == self.batch_size:
                batch_data.append(collate.default_collate(data))
                data.clear()
            data.append(d)
        if data:
            batch_data.append(collate.default_collate(data))
            data.clear()

        meta = {'duration': audio_len, 'sample_rate': self.sample_rate, 'audio_file': file}

        return batch_data, meta, intervals


class MusicLocalFeature:
    def __init__(self,
                 model_path: str,
                 sample_rate=16000,
                 feature_sample_rate=1,
                 sample_len=8,
                 device=torch.device("cpu"),
                 beat_sync=False,
                 batch_size=1024):
        self.sample_rate = sample_rate
        self.sample_len = sample_len
        self.beat_sync = beat_sync
        self.downbeat_sync = not beat_sync
        if self.beat_sync:
            raise NotImplementedError
        self.batch_size = batch_size
        self.device = device

        checkpoint = torch.load(model_path, map_location=self.device)

        self.chunk_mode = checkpoint['chunk_mode']
        self.model = checkpoint['model_select']

        torch.manual_seed(0)
        self.frontend_model = HarmonicCNN(sample_rate=self.sample_rate, n_class=dim_audio_fea)
        self.backend_model = metric_mlp(input_dim=dim_audio_fea, output_dim=dim_metric)
        self.frontend_model.load_state_dict(checkpoint['frontend_model_state_dict'])
        self.backend_model.load_state_dict(checkpoint['backend_model_state_dict'])
        self.frontend_model.to(self.device)
        self.backend_model.to(self.device)
        if torch.device.type == "cuda":
            self.frontend_model = torch.nn.DataParallel(self.frontend_model)
            self.backend_model = torch.nn.DataParallel(self.backend_model)
        self.frontend_model.eval()
        self.backend_model.eval()

        self.single_audio_prepared_data = []

        self.audio_files = []
        self.feature_sample_rate = feature_sample_rate

    def run(self, num_workers=0):
        raise NotImplementedError

    def __len__(self):
        return len(self.audio_files)

    def append_task(self, audio_file, *args, **kwargs):
        self.audio_files.append(audio_file)


class MusicLocalFeatureSynchronized(MusicLocalFeature):
    """
    match the local feature with the transition
    """

    def __init__(self,
                 model_path: str,
                 sample_rate=16000,
                 feature_sample_rate=1,
                 sample_len=8,
                 device=torch.device("cpu"),
                 beat_sync=False,
                 batch_size=1024):
        super(MusicLocalFeatureSynchronized, self).__init__(model_path=model_path,
                                                            sample_rate=sample_rate,
                                                            feature_sample_rate=feature_sample_rate,
                                                            sample_len=sample_len,
                                                            device=device,
                                                            beat_sync=beat_sync,
                                                            batch_size=batch_size)
        self.at_times = []

    def append_task(self, audio_file, transitions=None, *args, **kwargs):
        assert transitions is not None
        self.audio_files.append(audio_file)
        self.at_times.append(self.extract_at_time(transitions))

    @staticmethod
    def extract_at_time(transition_annotations):
        at_time = []
        transition_annotations = sorted(transition_annotations, key=lambda x: x["start"])
        for t in transition_annotations:
            start = t["start"]
            end = t["end"]
            start /= 1000
            end /= 1000
            start = round(start, 2)
            end = round(end, 2)

            at_time.append(start)
            at_time.append(end)
        return at_time

    @torch.no_grad()
    def run(self, num_workers=0):
        ds = ExtractDataloaderSet(self.audio_files, feature_sample_rate=self.feature_sample_rate,
                                  sample_rate=self.sample_rate, batch_size=self.batch_size, args=self,
                                  at_time=self.at_times)
        dl = DataLoader(ds,
                        collate_fn=lambda x: x,
                        num_workers=num_workers,
                        batch_size=1,
                        prefetch_factor=2)

        for d in dl:
            try:
                assert len(d) == 1  # make sure bs=1
                iter_data, meta, intervals = d[0]  # get first batch of data
                if iter_data is None:
                    yield None
                    continue
                embeds = torch.empty(0, dim_metric).to(self.device)
                for sample, label, beat_time in iter_data:
                    audio_fea = self.frontend_model(sample.float().to(self.device))
                    embed = l2_norm(self.backend_model(audio_fea))
                    embeds = torch.cat((embeds, embed), 0)
                embeds = embeds.cpu().numpy().tolist()

                result = {'metadata': meta}
                out_embeds = {}
                for i in range(len(intervals)):
                    out_embeds[intervals[i][1]] = embeds[i]
                result['embed'] = out_embeds
                yield result
            except Exception as e:
                logger.warning("Extractor received failure while processing a sample: {}".format(e))
                yield None


class AudioSamples(Dataset):

    def __init__(self, samples, labels, args):
        self.downbeat_sync = args.downbeat_sync
        self.sample_rate = args.sample_rate
        self.sample_len_sec = args.sample_len
        self.sample_len = args.sample_rate * self.sample_len_sec
        self.samples = samples
        self.chunk_mode = args.chunk_mode
        uniq_labels = list(set([l[2] for l in labels]))
        self.label_map = {}
        for i in range(len(uniq_labels)):
            self.label_map[uniq_labels[i]] = i
        self.labels = [(l[0], l[1], self.label_map[l[2]]) for l in labels]
        self.n_sample = len(self.labels)
        self.indices = np.array(range(self.n_sample))

    def __len__(self):
        return self.n_sample

    def __getitem__(self, index):
        times, mid_time, label = self.labels[self.indices[index]]
        if self.chunk_mode == "front":
            start = int(round(times[0] * self.sample_rate))
            x = self.samples[start: start + self.sample_len]
        elif self.chunk_mode == "alone":
            x = self.samples[int(round(times[0] * self.sample_rate)):
                             int(round(times[1] * self.sample_rate))]
        elif "center" in self.chunk_mode:
            middle = int(round(mid_time * self.sample_rate))
            start = middle - (self.sample_len // 2)
            end = middle + (self.sample_len // 2)
            x = self.samples[max(0, start): min(len(self.samples), end)]
            x = np.pad(x, (max(0, -start), max(end - len(self.samples), 0)), 'constant')
            x = x[:self.sample_len]
            if self.chunk_mode == "centerwin":
                beat_dur = times[1] - times[0]
                sustain_len = int(beat_dur * self.sample_rate)
                if not self.downbeat_sync:
                    sustain_len *= 4
                if sustain_len < self.sample_len - 100:
                    ramp_len = self.sample_len - sustain_len
                    ramp = np.hamming(ramp_len)
                    w = np.concatenate((ramp[:ramp_len // 2], np.ones(sustain_len), ramp[ramp_len // 2:]))
                    x = np.multiply(x, w)
        elif self.chunk_mode == "alone":
            x = self.samples[int(round(times[0] * self.sample_rate)):
                             int(round(times[1] * self.sample_rate))]
        else:
            print("Cannot determine the chunk mode...")
            exit()

        if len(x) == self.sample_len:
            pass
        elif len(x) > self.sample_len:
            x = x[: self.sample_len]
        else:
            x = np.pad(x, (0, self.sample_len - len(x)), 'constant')

        return x, label, times[0]
