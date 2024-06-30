# https://stackoverflow.com/questions/33879523/python-how-can-i-generate-a-wav-file-with-beeps
import numpy as np
import wave
import struct

from quantumreservoirpy.util import listify


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
    arr = (
        (num_samples - linspc)
        / num_samples
        * np.sin(2 * np.pi * freq * (linspc / sample_rate))
    )

    return arr


def composite(freqs, duration_milliseconds):
    tone = sinewave(freqs[0], duration_milliseconds=duration_milliseconds)
    for freq in freqs[1:]:
        tone += sinewave(freq=freq, duration_milliseconds=duration_milliseconds)
    return tone / max(max(abs(tone)), 1)


def gen_audio(noter, filename="output.wav", BPM=144):
    DUR = 60000 / BPM * 4

    # wav_file=wave.open(filename,"w")

    with wave.open(filename, "w") as wav_file:
        # num = 0
        nchannels = 1
        sampwidth = 2
        # nframes = len(audio)
        comptype = "NONE"
        compname = "not compressed"
        wav_file.setparams((nchannels, sampwidth, sample_rate, 0, comptype, compname))

        for midi, duration in noter:
            dur = DUR * duration
            if midi == "P":
                arr = silence(duration_milliseconds=dur)
            else:
                freqs = [midi_to_frequency(mid) for mid in listify(midi)]
                arr = composite(freqs, duration_milliseconds=dur)
            # num += len(arr)
            for sample in arr:
                wav_file.writeframes(struct.pack("h", int(sample * 32767.0)))
        # wav_file.setnframes(num)
