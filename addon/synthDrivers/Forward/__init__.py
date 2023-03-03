from collections import OrderedDict
from synthDriverHandler import SynthDriver
from speech.commands import IndexCommand, RateCommand, PitchCommand
from synthDrivers.Forward import client

class SynthDriver(SynthDriver):
	"""ForwardTacotron For NVDA.
	Author: Mateo C"""
	name="Forward"
	# Translators: Description for the speech synthesizer.
	description=_("ForwardTacotron")

	@classmethod
	def check(cls):
		return True

	supportedSettings = (SynthDriver.RateSetting(),SynthDriver.PitchSetting(), SynthDriver.InflectionSetting())
	supportedCommands = {RateCommand, PitchCommand}
	#_availableVoices = OrderedDict({name: synthDriverHandler.VoiceInfo(name, description)})

	def __init__(self):
		self.rate=1 # normal rate: 1.0
		self.pitch=1
		self.energy=1
		self.factor = 2 # maximum rate/pitch/energy (in 2.0x).
		self.vocoder="hifigan" # hifigan, griffinlim. A combo box could be made to select it in a future release.
		self.freq=22050 # sample rate for HiFi-GAN inference. 22050 and 32000 are currently supported.

	PROSODY_ATTRS = {RateCommand: "rate", PitchCommand: "pitch"}

	# These are functions to convert from percent to x, x to percent, and vice versa. ForwardTacotron uses the parameters based on x.
	def percent_to_x(self, percent, factor):
		return (percent / 100) * factor

	def x_to__percent(self, x, factor):
		return (x/factor) * 100

	def _get_rate_percent(self):
		return self.x_to__percent(self.rate, 2)

	def _get_rate(self):
		return self._get_rate_percent

	def _get_pitch_percent(self):
		return self.x_to__percent(self.pitch, 2)

	def _get_pitch(self):
		return self._get_pitch_percent

	def _set_rate(self,rate):
		self.Rate = self.prosody_percent_to_x(rate, 2)

	def _set_pitch(self,pitch):
		self._pitch=self.prosody_percent_to_x(pitch, 2)

	def speak(self, speechSequence):
		self.lastIndex = None
		for item in speechSequence:
			if isinstance(item, IndexCommand):
				self.lastIndex = item.index
				client.send_text(speechSequence, vocoder = self.vocoder, freq = self.freq, rate = self.rate, pitch = self.pitch, energy = self.energy)
			elif type(item) in self.PROSODY_ATTRS:
				if type(item) == RateCommand:
					print(item)
				if type(item) == PitchCommand:
					print(item)
	def cancel(self):
		self.lastIndex = None

	def _get_voice(self):
		return self.name
