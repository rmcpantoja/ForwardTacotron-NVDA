# [ForwardTacotron](https://github.com/as-ideas/ForwardTacotron) and [HiFi-GAN](https://github.com/jik876/hifi-gan) support for NVDA

**Note: This add-on as well as the documentation is still under construction. Your contributions are welcome!**

## to do:

- [x] A way to compile and integrate the server to the plugin.
	- [x] When this happens, allow the server to open when the synth is loaded. Once the server loads, we can call check to make the speech synthesizer ready for use.
	- [x] Two versions could be made for the add-on, with CPU support and one with GPU support, since apparently the synthesis is generated in real time on a GPU. In the meantime, we may notice slowdowns in the CPU.
- [x] Voice, rate, intonation and energy change support in synth ring options.
- [x] At the moment the plugin uses httplib2 to communicate with the server, but I could look for other methods and if necessary rewrite a part of the server.
- [x] Add support for loading different voices that could be detected within a "voice_models" folder.
- [x] For newer multi-speaker models, it can read the settings to check, and if so, it can choose the voice from the synth ring options with first consult the speaker names on the model.
