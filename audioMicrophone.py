import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pyaudio
from matplotlib.colors import LogNorm
from matplotlib.mlab import window_hanning,specgram

class spectrogram:
    fig = plt.figure()            
    pa = pyaudio.PyAudio()
    audioFormat = pyaudio.paInt16

    def __init__(self, chunkSize, sampleRate, sampleLength, channels, samplesPerFrame, nfft, hop):
        self.chunkSize = chunkSize ## samples per read        
        self.sampleRate = sampleRate
        self.sampleLength = int(chunkSize*1000/sampleRate) # in ms
        self.channels = channels # audio channels for microphone
        self.samplesPerFrame = samplesPerFrame
        self.nfft = nfft
        self.hop = hop        

       
    def readMicrophone(self):                
        stream = self.pa.open(format = self.audioFormat,
                         channels = self.channels,
                         rate = self.sampleRate,
                         input = True,
                         frames_per_buffer = self.chunkSize)        
        return stream

    def getData(self, stream):
        inputData = stream.read(self.chunkSize)
        self.data = np.fromstring(inputData,np.int16) 
        return self.data       

    def getSpectrogram(self, data):        
        arr2D,freqs,bins = specgram(data, window=window_hanning,
                                    Fs = self.sampleRate, NFFT=self.nfft, noverlap=self.hop)
        return arr2D,freqs,bins

    def showSpectrogram(self):        
        stream = self.readMicrophone()
        self.stream = stream
        arr2D,freqs,bins = specgram(self.getData(stream),window=window_hanning,
                                    Fs = self.sampleRate, NFFT=self.nfft, noverlap=self.hop)
        extent = (bins[0], bins[-1] * self.samplesPerFrame, freqs[-1], freqs[0])
        self.im = plt.imshow(arr2D,aspect="auto",extent = extent,interpolation="none",
                            cmap = "jet",norm = LogNorm(vmin=.01,vmax=1))
        plt.xlabel("Time (seconds)")
        plt.ylabel("Frequency (Hz)")
        plt.title("Streaming Spectrogram")
        plt.gca().invert_yaxis()
        ##plt.colorbar() #enable if you want to display a color bar            
        anim = animation.FuncAnimation(self.fig, self.updateFig, blit = False,
                                           interval=self.chunkSize/1000)    
        try:
            plt.show()
        except:
            print("** plot closed")               
        stream.stop_stream()
        stream.close()
        self.pa.terminate()
        print("** program closed")

    def updateFig(self, n):
        data = self.getData(self.stream)        
        arr2D,freqs,bins = self.getSpectrogram(data)
        im_data = self.im.get_array()
        if n < self.samplesPerFrame:
            im_data = np.hstack((im_data,arr2D))
            self.im.set_array(im_data)
        else:
            keep_block = arr2D.shape[1]*(self.samplesPerFrame - 1)
            im_data = np.delete(im_data,np.s_[:-keep_block],1)
            im_data = np.hstack((im_data,arr2D))
            self.im.set_array(im_data)
        return self.im,