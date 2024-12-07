import os
import urllib.request

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import Wav2Vec2MOS
import warnings


warnings.filterwarnings("ignore", category=UserWarning)

class MosMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        path = 'wmos_pretrained/wv_mos.ckpt'

        if (not os.path.exists(path)):
            print("Downloading the checkpoint for WV-MOS")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            urllib.request.urlretrieve(
                "https://zenodo.org/record/6201162/files/wav2vec2.ckpt?download=1",
                path
            )
            print('Weights downloaded in: {} Size: {}'.format(path, os.path.getsize(path)))


        self.model = Wav2Vec2MOS(path)
        self.mos = []

    def __call__(self, generated_wavs,initial_lens=None, **kwargs):
        if initial_lens is None:
            for audio in generated_wavs:
                self.mos.append(self.model.calculate_one(audio))
        else:
            tuples = list(zip(generated_wavs, initial_lens))
            for audio, true_len in tuples:
                self.mos.append(self.model.calculate_one(audio[:, :true_len]))

