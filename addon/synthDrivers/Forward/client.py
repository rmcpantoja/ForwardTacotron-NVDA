from synthDrivers.Forward._contrib import httplib2
import urllib
http_obj = httplib2.Http()
forward_TTS_url = 'http://localhost:5000/synthesize'

def send_text(speechSequence, rate, pitch, energy, vocoder, freq):
	text = speechSequence[0]
	# Define request parameters. (json format.)
	params = {'text': text, 'alpha': rate, 'pitch': pitch, 'energy': energy, 'vocoder': vocoder, 'freq': freq}
	encoded_params = urllib.parse.urlencode(params)
	# make request to server:
	resp, content = http_obj.request(uri=forward_TTS_url + '?' + encoded_params, method='GET')
