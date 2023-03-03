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

if not os.path.exists("forward_step90k.pt"):
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
from builtins import int

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
    if not exists(hifigan_pretrained_model):
        if MODEL_ID == 1:
          gdown.download("https://github.com/justinjohn0306/tacotron2/releases/download/assets/Superres_Twilight_33000", hifigan_pretrained_model, quiet = False)
        elif MODEL_ID == "universal":
          gdown.download("https://github.com/justinjohn0306/tacotron2/releases/download/assets/g_02500000", hifigan_pretrained_model, quiet = False)
        else:
          gdown.download(d+MODEL_ID, hifigan_pretrained_model, quiet=False)
    # check for HiFi-GAN, if the model isn't downloaded definitely:
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
checkpoint_path = "forward_step90k.pt"
vocoder_checkpoint_path = "hifi_pretrained.pt"
tts_model, config = load_tts_model(checkpoint_path)
dsp = DSP.from_config(config)
if not os.path.exists("/contnet/hifimodel_config_v1"):
  hifigan, h, denoiser = get_hifigan(vocoder_id, "config_v1")
  hifigan_sr, h2, denoiser_sr = get_hifigan(1, "config_32k")
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
    freq = int(request.args.get('freq'))
    if not freq:
        freq = 32000 # 22050, 32000
    vocoder = request.args.get('vocoder')
    if not vocoder:
        vocoder = 'hifigan' # hifigan, griffinlim
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
    if vocoder == "hifigan":
        m = m.cpu()
        with torch.no_grad():
            y_g_hat = hifigan(m)
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio_denoised = denoiser(audio.view(1, -1), strength=35)[:, 0]
            # Resample to 32k
            audio_denoised = audio_denoised.cpu().numpy().reshape(-1)
            normalize = (MAX_WAV_VALUE / np.max(np.abs(audio_denoised))) ** 0.9
            audio_denoised = audio_denoised * normalize
            if freq == 32000:
                wave = resampy.resample(
                    audio_denoised,
                    h.sampling_rate,
                    h2.sampling_rate,
                    filter="sinc_window",
                    window=scipy.signal.windows.hann,
                    num_zeros=8,
                )
                wave_out = wave.astype(np.int16)
                # HiFi-GAN super-resolution
                wave = wave / MAX_WAV_VALUE
                wave = torch.FloatTensor(wave).to(torch.device("cpu"))
                new_mel = mel_spectrogram(
                    wave.unsqueeze(0),
                    h2.n_fft,
                    h2.num_mels,
                    h2.sampling_rate,
                    h2.hop_size,
                    h2.win_size,
                    h2.fmin,
                    h2.fmax,
                )
                y_g_hat2 = hifigan_sr(new_mel)
                audio2 = y_g_hat2.squeeze()
                audio2 = audio2 * MAX_WAV_VALUE
                audio2_denoised = denoiser(audio2.view(1, -1), strength=35)[:, 0]
                # High-pass filter, mixing and denormalizing
                audio2_denoised = audio2_denoised.cpu().numpy().reshape(-1)
                b = scipy.signal.firwin(
                    101, cutoff=10500, fs=h2.sampling_rate, pass_zero=False
                )
                y = scipy.signal.lfilter(b, [1.0], audio2_denoised)
                y *= 1.0
                y_out = y.astype(np.int16)
                y_padded = np.zeros(wave_out.shape)
                y_padded[: y_out.shape[0]] = y_out
                sr_mix = wave_out + y_padded
                sr_mix = sr_mix / normalize
                finalaudio = sr_mix.astype(np.int16)
            elif freq == 22050:
                finalaudio = audio_denoised.astype(np.int16)
            else:
                raise Exception("This sample rate doesn't supported. 22050 and 32000 only.")
    elif vocoder == "griffinlim":
        freq = 22050
        finalaudio = dsp.griffinlim(m.squeeze().numpy())
    else:
        raise Exception("This vocoder doesn't supported. Only hifigan and griffinlim.")
    # play audio
    sd.play(finalaudio, freq)
    return send_file(io.BytesIO(finalaudio), attachment_filename="synthesized.wav", mimetype="audio/wav")

if __name__ == "__main__":
    app.run(debug=False)