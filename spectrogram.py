import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LogNorm
from matplotlib.mlab import window_hanning,specgram

class Spectrogram:
    fig = plt.figure()
    pa = pyaudio.PyAudio()
    audio_format = pyaudio.paInt16
    chunk_size = 8192 # samples per read
    sample_rate = 44100
    sample_length = int(chunk_size*1000/sample_rate) # in ms
    channels = 1 # audio channels for microphone
    samples_per_frame = 10
    nfft = 1024
    hop = 512

    def open_microphone(self):
        stream = self.pa.open(format = self.audio_format,
            channels = self.channels,
            rate = self.sample_rate,
            input = True,
            frames_per_buffer = self.chunk_size)
        return stream

    def get_data(self, stream):
        #gets microphone data
        input_data = stream.read(self.chunk_size)
        data = np.fromstring(input_data,np.int16) 
        return data

    def get_spectrogram(self, data):
        arr2D,freqs,bins = specgram(data,
            window=window_hanning,
            Fs = self.sample_rate,
            NFFT=self.nfft,
            noverlap=self.hop)
        return arr2D,freqs,bins

    def update_figure(self, n):
        data = self.get_data(self.stream)
        arr2D,freqs,bins = self.get_spectrogram(data)
        im_data = self.im.get_array()
        if n < self.samples_per_frame:
            im_data = np.hstack((im_data,arr2D))
            self.im.set_array(im_data)
        else:
            keep_block = arr2D.shape[1]*(self.samples_per_frame - 1)
            im_data = np.delete(im_data,np.s_[:-keep_block],1)
            im_data = np.hstack((im_data,arr2D))
            self.im.set_array(im_data)
        return self.im,

    def stream_spectrogram(self):
        try:
            print("[Launching Streaming]")
            self.stream = self.open_microphone()
            arr2D,freqs,bins = specgram(self.get_data(self.stream),
                window=window_hanning,
                Fs = self.sample_rate,
                NFFT=self.nfft,
                noverlap=self.hop)
            extent = (bins[0], bins[-1] * self.samples_per_frame, freqs[-1], freqs[0])
            self.im = plt.imshow(arr2D,
                aspect="auto",
                extent = extent,
                interpolation="none",
                cmap = "jet",
                norm = LogNorm(vmin=.01,vmax=1))
            plt.xlabel("Time (seconds)")
            plt.ylabel("Frequency (Hz)")
            plt.title("Streaming Spectrogram")
            plt.gca().invert_yaxis()
            plt.colorbar() #enable if you want to display a color bar
            anim = animation.FuncAnimation(self.fig,
                self.update_figure,
                blit = False,
                interval=self.chunk_size/1000)
            plt.show()
            print()
            print("[Stopping Streaming]")
        except:
            self.stream.stop_stream()
            self.stream.close()
            self.pa.terminate()
            print("[Program Closed]")
            