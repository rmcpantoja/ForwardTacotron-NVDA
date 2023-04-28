# [ForwardTacotron](https://github.com/as-ideas/ForwardTacotron) and [HiFi-GAN](https://github.com/jik876/hifi-gan) support for [NVDA Screen reader](https://github.com/nvaccess/nvda)

**Note: This add-on as well as the documentation is still under construction. Your contributions are welcome!**

## introduction

Remember that ForwardTacotron is a speech synthesis model in pytorch that uses a duration predictor to align text and generated mel spectrograms. The model has advantages, such as robustness, speed, pitch and energy manipulation, and efficiency.

So, this plugin is an attempt to implement support for ForwardTacotron in NVDA's open source screen reader via client/server, because the libraries used as torch are not possible to include in NVDA directly.

This is a work in progress and therefore there is still a lot to do.

In the meantime, you can listen to the progress that has been made so far.

### audio samples

| Language | Voice | Sample |
|:---:|:---:|:---:|
|English|[LJSpeech](https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2) (with griffinLim vocoder)|<audio src="https://github.com/rmcpantoja/ForwardTacotron-NVDA/raw/main/demo/ForwardTacotron%20NVDA%20ljspeech%20griffinlim.mp3?raw=true" controls preload></audio>|
|English|[LJSpeech](https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2) (with HiFi-GAN vocoder)|<audio src="https://github.com/rmcpantoja/ForwardTacotron-NVDA/raw/main/demo/ForwardTacotron NVDA ljspeech hifigan.mp3?raw=true" controls preload></audio>|
|Spanish|[Ald Dataset](https://huggingface.co/datasets/rmcpantoja/Ald_Mexican_Spanish_speech_dataset) (with HiFi-GAN vocoder)|<audio src="https://github.com/rmcpantoja/ForwardTacotron-NVDA/raw/main/demo/ForwardTacotron NVDA Spanish Ald hifigan.mp3?raw=true" controls preload></audio>|
|Spanish|Odal (with HiFi-GAN vocoder, universal model)|<audio src="https://github.com/rmcpantoja/ForwardTacotron-NVDA/raw/main/demo/ForwardTacotron NVDA Spanish Odal hifigan (universal).mp3?raw=true" controls preload></audio>|

## to do:

- [x] A way to compile and integrate the server to the add-on.
	- [x] When this happens, allow the server to open when the synth is loaded. Once the server loads, we can call check to make the speech synthesizer ready for use.
	- [x] Two versions could be made for the add-on, with CPU support and one with GPU support, since apparently the synthesis is generated in real time on a GPU. In the meantime, we may notice slowdowns in the CPU.
- [x] Voice and energy change support in synth ring options.
- [x] At the moment the add-on uses httplib2 to communicate with the server, but I could look for other methods and if necessary rewrite a part of the server.
- [x] Add support for loading different voices that could be detected within a "voice_models" folder.
	- [x] With this, a support for downloading trained models could be added. We have a ljspeech model in English, another in German and two in Spanish.
- [x] For newer multi-speaker models, it can read the settings to check, and if so, it can choose the voice from the synth ring options with first consult the speaker names on the model.
