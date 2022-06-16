from pydub import AudioSegment
from config import config
import numpy as np
import webrtcvad
import soundfile as sf
from python_speech_features import mfcc, delta


def split_frames(data):
    frames = np.array(np.array_split(data, config.frame_size))
    # print(frames.shape)
    return frames.T

def vad(model, filepath):
    model.eval()
    audio_b = (AudioSegment.from_file(filepath)
                         .set_frame_rate(config.sr)
                         .set_sample_width(2)
                         .set_channels(1))
    audio_b = np.array(audio_b.get_array_of_samples(), dtype=np.int16)
    vad = webrtcvad.Vad(0)

    label_webrtc = [1 if vad.is_speech(f.tobytes(), sample_rate=config.sr) else 0 for f in
                   split_frames(audio_b)]

    audio, _ = sf.read(filepath)
    audio_frames = split_frames(audio)
    audio_mfcc = mfcc(audio, samplerate=config.sr, winlen=2 * config.frame_len / 1000,
                    winstep=config.frame_len / 1000,
                    nfft=2048)
    audio_mfcc = audio_mfcc[:, 1:]
    audio_delta = delta(audio_mfcc, 1)

