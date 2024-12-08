from transformers import Wav2Vec2Model, Wav2Vec2Processor
import torch    
from collections import OrderedDict
from torch import nn
import torchaudio

def extract_prefix(prefix, weights):
    result = OrderedDict()
    for key in weights:
        if key.find(prefix) == 0:
            result[key[len(prefix):]] = weights[key]
    return result     

    
class Wav2Vec2MOS(nn.Module):
    def __init__(self, path, freeze=True):
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.freeze = freeze
        
        self.dense = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        
        if self.freeze:
            self.encoder.eval()
            for p in self.encoder.parameters():
                p.requires_grad_(False)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.load_state_dict(extract_prefix('model.', torch.load(path, map_location=self.device)['state_dict']))
        self.eval()
        # self.cuda()
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        
    def forward(self, x):
        x = self.encoder(x)['last_hidden_state'] # [Batch, time, feats]
        x = self.dense(x) # [batch, time, 1]
        x = x.mean(dim=[1,2], keepdims=True) # [batch, 1, 1]
        return x
                
        
    def calculate_one(self, wav_file):
        wav_file_new = torch.clone(wav_file)
        wav_file_new = wav_file_new.squeeze(0)
        x = torchaudio.functional.resample(wav_file_new, 22050, 16000)
        x = self.processor(x, return_tensors="pt", padding=True, sampling_rate=16000).input_values
        with torch.no_grad():
            res = self.forward(x).mean()
        return res.cpu().item()
    

    