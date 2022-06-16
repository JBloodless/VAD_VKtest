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

batch_size = 128
features = 24
val_size = 256
train_size = 1024
test_size = 128

noise_snrs = [-5, 0, 5, 10]

temp_folder = r'D:\vk_test\VAD\temp'