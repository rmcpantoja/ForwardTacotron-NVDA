from collections import OrderedDict
import synthDriverHandler
from speech.commands import IndexCommand
from synthDrivers.Forward._contrib import httplib2
import urllib
http_obj = httplib2.Http()
forward_TTS_url = 'http://localhost:5000/synthesize'
class SynthDriver(synthDriverHandler.SynthDriver):
	"""FT synth.
	"""
	name="Forward"
	# Translators: Description for a speech synthesizer.
	description=_("ForwardTacotron")

	@classmethod
	def check(cls):
		return True

	supportedSettings = frozenset()
	_availableVoices = OrderedDict({name: synthDriverHandler.VoiceInfo(name, description)})

	def send_text(self, speechSequence):
		text = speechSequence[0]
		# Define request parameters
		params = {'text': text, 'alpha': 1.0, 'pitch': 1.0, 'energy': 1.0, 'vocoder': 'griffinlim', 'freq': 22050}
		encoded_params = urllib.parse.urlencode(params)
		# make request to server:
		resp, content = http_obj.request(uri=forward_TTS_url + '?' + encoded_params, method='GET')

	def speak(self, speechSequence):
		self.lastIndex = None
		for item in speechSequence:
			if isinstance(item, IndexCommand):
				self.lastIndex = item.index
				self.send_text(speechSequence)  # Change here
	def cancel(self):
		self.lastIndex = None

	def _get_voice(self):
		return self.name
