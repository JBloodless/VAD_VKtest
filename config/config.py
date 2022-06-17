import torch

sr = 16000

min_len_ms = 3000
max_len_ms = 10000

min_len = min_len_ms*sr//1000
max_len = max_len_ms*sr//1000

frame_len = 10

frame_size = frame_len*sr//1000

slice_min = min_len//frame_size
slice_max = max_len//frame_size

device = 'cuda' if torch.cuda.is_available() else 'cpu'

batch_size = 4
train_frames = 30
features = 24
val_size = 256
train_size = 1024
test_size = 528

epochs = 100

noise_snrs = [-5, 0, 5, 10]

train_sets = [r'D:\Datasets\LibriSpeech\train-clean-360', r'D:\Datasets\LibriSpeech\train-other-500']
val_sets = [r'D:\Datasets\LibriSpeech\train-clean-100']
test_sets = [r'D:\Datasets\LibriSpeech\dev_clean']
noise_sets = r'D:\Datasets\noises'

temp_folder = r'D:\vk_test\VAD\temp'