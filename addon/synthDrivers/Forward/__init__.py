from collections import OrderedDict
from synthDriverHandler import SynthDriver
from speech.commands import IndexCommand, RateCommand, PitchCommand
from synthDrivers.Forward import client

class SynthDriver(SynthDriver):
	"""ForwardTacotron For NVDA.
	Author: Mateo C"""
	name="Forward"
	description=_("ForwardTacotron")
	factor = 2 # maximum rate/pitch/energy (in 2.0x).

	@classmethod
	def check(cls):
		return True

	supportedSettings = (SynthDriver.RateSetting(),SynthDriver.PitchSetting(), SynthDriver.InflectionSetting())
	supportedCommands = {RateCommand, PitchCommand}
	#_availableVoices = OrderedDict({name: synthDriverHandler.VoiceInfo(name, description)})

	def __init__(self):
		super().__init__()
		self.rate_x=1 # normal rate: 1.0
		self.pitch_x=1
		self.energy_x=1
		self.vocoder="hifigan" # hifigan, griffinlim. A combo box could be made to select it in a future release.
		self.freq=22050 # sample rate for HiFi-GAN inference. 22050 and 32000 are currently supported.

	PROSODY_ATTRS = {RateCommand: "rate", PitchCommand: "pitch"}

	# These are functions to convert from percent to x, x to percent, and vice versa. ForwardTacotron uses the parameters based on x.
	def percent_to_x(self, percent):
		return (percent / 100) * self.factor

	def x_to_percent(self, x):
		return int(x/self.factor * 100)

	def _get_rate(self):
		return self.rate_x

	def _get_pitch(self):
		return self.pitch_x

	def _set_rate(self, rate):
		self.rate_x = self.percent_to_x(rate)
		print(f"rate set to: {self.rate_x}")

	def _set_pitch(self,pitch):
		self.pitch_x=self.percent_to_x(pitch)
		print(f"Pitch set to: {self.pitch_x}")

	def speak(self, speechSequence):
		self.lastIndex = None
		self.rate = self.rate_x
		self.pitch = self.pitch_x
		self.energy= self.energy_x
		for item in speechSequence:
			if isinstance(item, IndexCommand):
				self.lastIndex = item.index
			elif type(item) in self.PROSODY_ATTRS:
				if isinstance(item, RateCommand):
					self.rate = self.percent_to_x(item.value)
				elif isinstance(item, PitchCommand):
					self.pitch = self.percent_to_x(item.value)
		client.send_text(speechSequence, vocoder=self.vocoder, freq=self.freq, rate=self.rate, pitch=self.pitch, energy=self.energy)

	def cancel(self):
		self.lastIndex = None

	def _get_voice(self):
		return self.name
