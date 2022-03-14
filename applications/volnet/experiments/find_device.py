import argparse

import torch

parser = argparse.ArgumentParser()
parser.add_argument('--i', type=str, help='dummy variable', default='0')
args = vars(parser.parse_args())
dummy = args['i']
print(f'[INFO] Dummy: {dummy}')

cuda = torch.device('cuda')
all_devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
a = torch.randn(10, device=cuda)
print([torch.randn(10, device=cuda).device == device for device in all_devices])
