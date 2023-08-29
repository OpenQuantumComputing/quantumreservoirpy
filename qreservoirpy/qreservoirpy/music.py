# https://stackoverflow.com/questions/33879523/python-how-can-i-generate-a-wav-file-with-beeps
import numpy as np
import wave
import struct

from .utilities import listify
# Audio will contain a long list of samples (i.e. floating point numbers describing the
# waveform).  If you were working with a very long sound you'd want to stream this to
# disk instead of buffering it all in memory list this.  But most sounds will fit in
# memory.

def midi_to_frequency(note):
    a = 440
    return (a / 32) * (2 ** ((note - 9) / 12))

sample_rate = 44100.0

def silence(duration_milliseconds=500):
    num_samples = duration_milliseconds * (sample_rate / 1000.0)
    return np.zeros(int(num_samples))


def sinewave(freq=440.0, duration_milliseconds=500):

    num_samples = duration_milliseconds * (sample_rate / 1000.0)

    linspc = np.arange(int(num_samples))


    arr = (num_samples-linspc)/num_samples * np.sin(2 * np.pi * freq * ( linspc / sample_rate ))

    return arr

def composite(freqs, duration_milliseconds):
    tone = sinewave(freqs[0], duration_milliseconds=duration_milliseconds)
    for freq in freqs[1:]:
        tone += sinewave(freq=freq, duration_milliseconds=duration_milliseconds)
    return tone / max(max(abs(tone)), 1)


def save_wav(file_name, audio):
    # Open up a wav file
    wav_file=wave.open(file_name,"w")

    # wav params
    nchannels = 1

    sampwidth = 2

    # 44100 is the industry standard sample rate - CD quality.  If you need to
    # save on file size you can adjust it downwards. The stanard for low quality
    # is 8000 or 8kHz.
    nframes = len(audio)
    comptype = "NONE"
    compname = "not compressed"
    wav_file.setparams((nchannels, sampwidth, sample_rate, nframes, comptype, compname))

    # WAV files here are using short, 16 bit, signed integers for the
    # sample size.  So we multiply the floating point data we have by 32767, the
    # maximum value for a short integer.  NOTE: It is theortically possible to
    # use the floating point -1.0 to 1.0 data directly in a WAV file but not
    # obvious how to do that using the wave module in python.
    for sample in audio:
        wav_file.writeframes(struct.pack('h', int( sample * 32767.0 )))

    wav_file.close()

    return


def gen_audio(noter, filename="output.wav", BPM=144):
    DUR = 60000 / BPM * 4
    audio = []

    for midi, duration in noter:
        dur = DUR * duration
        if midi == 'P':
            arr = silence(duration_milliseconds=dur)
        else:
            freqs = [midi_to_frequency(mid) for mid in listify(midi)]
            arr = composite(freqs, duration_milliseconds=dur)

        audio = np.append(audio, arr)



    save_wav(filename, audio)

