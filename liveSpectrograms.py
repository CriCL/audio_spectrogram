import audioMicrophone as objAm

chunkSize = 8192 ## samples per read
sampleRate = 44100
sampleLength = int(chunkSize*1000/sampleRate) # in ms
channels = 1 # audio channels for microphone
samplesPerFrame = 10
nfft = 1024
hop = 512

## creating audioMicrophone instance object
am = objAm.spectrogram(chunkSize, sampleRate, sampleLength, channels, samplesPerFrame, nfft, hop)
## launching streaming spectrogram    
am.showSpectrogram()