from pathlib import Path
from torch.utils.data import Dataset
import torchaudio
import random
from tqdm import tqdm

from src.model.melspec import MelSpectrogramConfig, MelSpectrogram


class HiFiGanDataset(Dataset):
    def __init__(self, data_path, limit=None, max_len=22528, **kwargs):
        data_path = Path(data_path)
        self.mel_creator = MelSpectrogram(MelSpectrogramConfig())
        self.wavs_and_paths = []
        self.max_len = max_len
        for file_path in tqdm((data_path).iterdir(), desc='Loading files'):
            wav, _ = torchaudio.load(file_path)
            wav = wav[0:1, :]
            path = file_path
            self.wavs_and_paths.append({'wav' : wav, 'path' : path})
        if limit is not None:
            self.wavs_and_paths = self.wavs_and_paths[:limit]

    def __len__(self):
        return len(self.wavs_and_paths)

    def __getitem__(self, idx):
        wav = self.wavs_and_paths[idx]['wav']
        path = self.wavs_and_paths[idx]['path']
        if self.max_len is not None:
            start = random.randint(0,  wav.shape[-1] - self.max_len)
            wav = wav[:, start : start + self.max_len]
        melspec = self.mel_creator(wav.detach()).squeeze(0)
        return {"wav": wav, 'path' : path, 'melspec' : melspec}
