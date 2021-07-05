import collections
import webrtcvad
import wave
import numpy as np
import contextlib
import struct


class Frame(object):
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


class Vad(object):

    def __init__(self, signal_path, aggressiveness):
        self.signal_path = signal_path
        self.aggressiveness = aggressiveness

    def read_wave(self, path):
        with contextlib.closing(wave.open(path, 'rb')) as wf:
            num_channels = wf.getnchannels()
            assert num_channels == 1
            sample_width = wf.getsampwidth()
            assert sample_width == 2
            sample_rate = wf.getframerate()
            assert sample_rate in (8000, 16000, 32000, 48000)
            pcm_data = wf.readframes(
                wf.getnframes())

            return pcm_data, sample_rate

    def frame_generator(self, frame_duration_ms, audio, sample_rate):

        n = int(sample_rate * (
                    frame_duration_ms / 1000.0) * 2)
        offset = 0
        timestamp = 0.0
        duration = (float(n) / sample_rate) / 2.0
        while offset + n < len(audio):
            yield Frame(audio[offset:offset + n], timestamp, duration)
            timestamp += duration
            offset += n

    def vad_collector(self, sample_rate, frame_duration_ms,
                      padding_duration_ms, vad, frames):

        num_padding_frames = int(padding_duration_ms / frame_duration_ms)
        ring_buffer = collections.deque(maxlen=num_padding_frames)

        triggered = False

        voiced_frames = []
        for frame in frames:
            is_speech = vad.is_speech(frame.bytes, sample_rate)

            if not triggered:
                ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])

                if num_voiced > 0.9 * ring_buffer.maxlen:
                    triggered = True

                    for f, s in ring_buffer:
                        voiced_frames.append(f)
                    ring_buffer.clear()
            else:

                voiced_frames.append(frame)
                ring_buffer.append((frame, is_speech))
                num_unvoiced = len([f for f, speech in ring_buffer if not speech])

                if num_unvoiced > 0.9 * ring_buffer.maxlen:
                    triggered = False
                    yield b''.join([f.bytes for f in voiced_frames])
                    ring_buffer.clear()
                    voiced_frames = []

        if voiced_frames:
            yield b''.join([f.bytes for f in voiced_frames])

    def get_speech_signal(self):
        audio, sample_rate = self.read_wave(self.signal_path)

        vad = webrtcvad.Vad(int(self.aggressiveness))
        frames = self.frame_generator(30, audio, sample_rate)
        frames = list(frames)
        segments = self.vad_collector(sample_rate, 30, 300, vad, frames)

        string = ""
        final_signal = bytearray(string, 'utf-8')

        for i, segment in enumerate(segments):
            final_signal.extend(segment)

        decimal_signal = np.empty(int(len(final_signal)/2), dtype=object)
        for i in range(len(final_signal)):
            [decimal_signal[i]] = struct.unpack('h', final_signal[2*i:2*i+2])
            if 2 * i + 2 == len(final_signal):
                break

        return decimal_signal, sample_rate