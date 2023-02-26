# download models and import required packages here ...
from flask import Flask, request, jsonify, send_file
import io
import sounddevice as sd
from os.path import exists, join, basename, splitext
import sys
sys.path.append('ForwardTacotron')
sys.path.append('hifi-gan')
import os
import gdown
d = 'https://drive.google.com/uc?id='
model_id = "15iPwatcdldZxq-kfUzmQcfdceKW0qKm-" #@param {type:"string"}
if not os.path.exists("pretrained.pt"):
  gdown.download(d+model_id, "pretrained.pt", quiet=False)
vocoder_id = "1-RuVOLZ94HhS27PRW0Dk9h-gcLHas7jn" #@param {type:"string"}
import os
from pathlib import Path
from typing import Tuple, Dict, Any, Union
import IPython.display as ipd
import numpy as np
import torch
import json
import resampy
import scipy.signal
from env import AttrDict
from meldataset import mel_spectrogram, MAX_WAV_VALUE
from hifimodels import Generator
from denoiser import Denoiser
from models.fast_pitch import FastPitch
from models.forward_tacotron import ForwardTacotron
from utils.checkpoints import init_tts_model
from utils.display import simple_table
from utils.dsp import DSP
from utils.files import read_config
from utils.paths import Paths
from utils.text.cleaners import Cleaner
from utils.text.tokenizer import Tokenizer


def load_tts_model(checkpoint_path: str) -> Tuple[Union[ForwardTacotron, FastPitch], Dict[str, Any]]:
    print(f'Loading tts checkpoint {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    config = checkpoint['config']
    tts_model = init_tts_model(config)
    tts_model.load_state_dict(checkpoint['model'])
    print(f'Initialized tts model: {tts_model}')
    print(f'Restored model with step {tts_model.get_step()}')
    return tts_model, config

def get_hifigan(MODEL_ID, conf_name):
    # Download HiFi-GAN
    hifigan_pretrained_model = 'hifimodel_' + conf_name
    #gdown.download(d+MODEL_ID, hifigan_pretrained_model, quiet=False)
    if MODEL_ID == 1:
      gdown.download("https://github.com/justinjohn0306/tacotron2/releases/download/assets/Superres_Twilight_33000", hifigan_pretrained_model, quiet = False)
    elif MODEL_ID == "universal":
      gdown.download("https://github.com/justinjohn0306/tacotron2/releases/download/assets/g_02500000", hifigan_pretrained_model, quiet = False)
    else:
      gdown.download(d+MODEL_ID, hifigan_pretrained_model, quiet=False)
    if not exists(hifigan_pretrained_model):
        raise Exception("HiFI-GAN model failed to download!")
    # Load HiFi-GAN
    conf = os.path.join("hifi-gan", conf_name + ".json")
    with open(conf) as f:
        json_config = json.loads(f.read())
    h = AttrDict(json_config)
    torch.manual_seed(h.seed)
    hifigan = Generator(h).to(torch.device("cpu"))
    state_dict_g = torch.load(hifigan_pretrained_model, map_location=torch.device("cpu"))
    hifigan.load_state_dict(state_dict_g["generator"])
    hifigan.eval()
    hifigan.remove_weight_norm()
    denoiser = Denoiser(hifigan, mode="normal")
    return hifigan, h, denoiser

# load models, tokenizer, and cleaner
checkpoint_path = "pretrained.pt"
vocoder_checkpoint_path = "hifi_pretrained.pt"
tts_model, config = load_tts_model(checkpoint_path)
dsp = DSP.from_config(config)
if not os.path.exists("/content/hifimodel_config_v1"):
  hifigan, h, denoiser = get_hifigan(vocoder_id, "config_v1")
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
tts_model.to(device)
cleaner = Cleaner.from_config(config)
tokenizer = Tokenizer()
print(f'Using device: {device}\n')
tts_k = tts_model.get_step() // 1000
tts_model.eval()


app = Flask(__name__)

@app.route('/synthesize', methods=["GET"])
def synthesize_route():
    text = request.args.get('text')
    if not text:
        return jsonify({"error": "no text"}), 400
    alpha = float(request.args.get('alpha'))
    if not alpha:
        alpha = 1
    pitch = float(request.args.get('pitch'))
    if not pitch:
        pitch = 1
    energy = float(request.args.get('energy'))
    if not energy:
        energy = 1
    pitch_function = lambda x: x * pitch
    energy_function = lambda x: x*energy
    x = cleaner(text)
    x = tokenizer(x)
    x = torch.as_tensor(x, dtype=torch.long, device="cpu").unsqueeze(0)
    gen = tts_model.generate(x=x,
                             alpha=alpha,
                             pitch_function=pitch_function,
                             energy_function=energy_function)
    m = gen['mel_post'].cpu()
    m = m.cpu()
    with torch.no_grad():
        y_g_hat = hifigan(m)
        audio = y_g_hat.squeeze()
        audio = audio * MAX_WAV_VALUE
        audio_denoised = denoiser(audio.view(1, -1), strength=35)[:, 0]
        audio_denoised = audio_denoised.cpu().numpy().reshape(-1)
        normalize = (MAX_WAV_VALUE / np.max(np.abs(audio_denoised))) ** 0.9
        audio_denoised = audio_denoised * normalize
        finalaudio = audio_denoised.astype(np.int16)
    # play audio
    sd.play(finalaudio.astype(np.int16), 22050)
    return send_file(io.BytesIO(finalaudio.astype(np.int16)), attachment_filename="synthesized.wav", mimetype="audio/wav")

if __name__ == "__main__":
    app.run(debug=False)