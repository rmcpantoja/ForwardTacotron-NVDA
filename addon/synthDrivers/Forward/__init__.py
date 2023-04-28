from collections import OrderedDict
from synthDriverHandler import SynthDriver
from speech.commands import IndexCommand, RateCommand, PitchCommand
from synthDrivers.Forward import client

class SynthDriver(SynthDriver):
	"""ForwardTacotron For NVDA.
	Author: Mateo C"""
	name="Forward"
	description=_("ForwardTacotron")
	min_factor = 0.5 # Minimum rate/pitch/energy.
	factor = 2 # maximum rate/pitch/energy (in 2.0x).

	@classmethod
	def check(cls):
		return True

	supportedSettings = (SynthDriver.RateSetting(),SynthDriver.PitchSetting(), SynthDriver.InflectionSetting())
	supportedCommands = {RateCommand, PitchCommand}
	#_availableVoices = OrderedDict({name: synthDriverHandler.VoiceInfo(name, description)})

	def __init__(self):
		self.rate=50 # 1.0x
		self.rate_x=1.5 # normal rate: 50%
		self.pitch=50 # pitch (1.0x)
		self.pitch_x=1.0
		self.energy_x=1.0
		self.inflection=50 # Energy behaves almost the same as inflection.
		self.vocoder="hifigan" # hifigan, griffinlim. A combo box could be made to select it in a future release.
		self.freq=22050 # sample rate for HiFi-GAN inference. 22050 and 32000 are currently supported.

	PROSODY_ATTRS = {RateCommand: "rate_x", PitchCommand: "pitch_x"}

	# These are functions to convert from percent to x, x to percent, and vice versa. ForwardTacotron uses the parameters based on x.
	def percent_to_x(self, percent):
		return (percent / 100) * self.factor

	def x_to_percent(self, x):
		return int(x/self.factor * 100)

	def speak(self, speechSequence):
		self.lastIndex = None
		for item in speechSequence:
			if isinstance(item, IndexCommand):
				self.lastIndex = item.index
			elif type(item) in self.PROSODY_ATTRS:
				if isinstance(item, RateCommand):
					self.rate = item.newValue
				elif isinstance(item, PitchCommand):
					self.pitch = self.percent_to_x(item.value)
		print(f"Rate: {self.rate}. Pitch: {self.pitch}. Energy: {self.inflection}")
		client.send_text(speechSequence, vocoder=self.vocoder, freq=self.freq, rate=self.rate, pitch=self.pitch, energy=self.energy_x)

# rate and pitch:
	def _get_rate(self):
		print(f"Rate Orig: {self.rate_x} rate.")
		return self.rate_x

	def _get_pitch(self):
		return self.pitch_x

	def _set_rate(self, rate):
		self.rate_x = rate
		print(f"After rate set to: {self.rate}.")

	def _set_pitch(self,pitch):
		self.pitch_x=pitch
		print(f"After Pitch set to: {self.pitch}.")

	def cancel(self):
		self.lastIndex = None

	def _get_voice(self):
		return self.name
