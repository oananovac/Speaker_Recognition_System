import collections
import webrtcvad
import wave
import sys
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
            assert num_channels == 1  # if condition, if it,s true nothing happens if false => exception
            sample_width = wf.getsampwidth()  # nr de biti pe care este scris un esantion
            assert sample_width == 2  # de obicei un esantion este scris pe 1 sau 2 biti la fisierele wav
            sample_rate = wf.getframerate()  # frecventa de esantionare
            assert sample_rate in (8000, 16000, 32000, 48000)
            pcm_data = wf.readframes(
                wf.getnframes())  # citeste si returneaza primele n frame uri // getnframe -- nr de frame uri din secv audio

            return pcm_data, sample_rate

    def frame_generator(self, frame_duration_ms, audio, sample_rate):
        """Generates audio frames from PCM audio data.
        Takes the desired frame duration in milliseconds, the PCM data, and
        the sample rate.
        Yields Frames of the requested duration.
        """
        n = int(sample_rate * (
                    frame_duration_ms / 1000.0) * 2)  # nr de esantioane =  rata de esantioanare(esantioane/s)  # se inmulteste cu 2 pt ca e nr de biti pe esantion
        offset = 0
        timestamp = 0.0
        duration = (float(n) / sample_rate) / 2.0  # durata de timp intre esantioane
        while offset + n < len(audio):
            yield Frame(audio[offset:offset + n], timestamp, duration)
            timestamp += duration
            offset += n

    def vad_collector(self, sample_rate, frame_duration_ms,
                      padding_duration_ms, vad, frames):
        """Filters out non-voiced audio frames.
        Given a webrtcvad.Vad and a source of audio frames, yields only
        the voiced audio.
        Uses a padded, sliding window algorithm over the audio frames.
        When more than 90% of the frames in the window are voiced (as
        reported by the VAD), the collector triggers and begins yielding
        audio frames. Then the collector waits until 90% of the frames in
        the window are unvoiced to detrigger.
        The window is padded at the front and back to provide a small
        amount of silence or the beginnings/endings of speech around the
        voiced frames.
        Arguments:
        sample_rate - The audio sample rate, in Hz.
        frame_duration_ms - The frame duration in milliseconds.
        padding_duration_ms - The amount to pad the window, in milliseconds.
        vad - An instance of webrtcvad.Vad.
        frames - a source of audio frames (sequence or generator).
        Returns: A generator that yields PCM audio data.
        """
        num_padding_frames = int(padding_duration_ms / frame_duration_ms)
        # We use a deque for our sliding window/ring buffer.
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
        # NOTTRIGGERED state.
        triggered = False

        voiced_frames = []
        for frame in frames:
            is_speech = vad.is_speech(frame.bytes, sample_rate)  # true sau false ( vorbire )


            sys.stdout.write('1' if is_speech else '0')
            if not triggered:
                ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])
                # If we're NOTTRIGGERED and more than 90% of the frames in
                # the ring buffer are voiced frames, then enter the
                # TRIGGERED state.
                if num_voiced > 0.9 * ring_buffer.maxlen:
                    triggered = True
                    sys.stdout.write('+(%s)' % (ring_buffer[0][0].timestamp,))
                    # We want to yield all the audio we see from now until
                    # we are NOTTRIGGERED, but we have to start with the
                    # audio that's already in the ring buffer.
                    for f, s in ring_buffer:
                        voiced_frames.append(f)
                    ring_buffer.clear()
            else:
                # We're in the TRIGGERED state, so collect the audio data
                # and add it to the ring buffer.
                voiced_frames.append(frame)
                ring_buffer.append((frame, is_speech))
                num_unvoiced = len([f for f, speech in ring_buffer if not speech])
                # If more than 90% of the frames in the ring buffer are
                # unvoiced, then enter NOTTRIGGERED and yield whatever
                # audio we've collected.
                if num_unvoiced > 0.9 * ring_buffer.maxlen:
                    sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                    triggered = False
                    yield b''.join([f.bytes for f in voiced_frames])
                    ring_buffer.clear()
                    voiced_frames = []
        if triggered:
            sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
        sys.stdout.write('\n')
        # If we have any leftover voiced audio when we run out of input,
        # yield it.
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
            decimal_signal[i] = struct.unpack('h', final_signal[2*i:2*i+2])[0]
            if 2 * i + 2 == len(final_signal):
                break

        return decimal_signal, sample_rate
