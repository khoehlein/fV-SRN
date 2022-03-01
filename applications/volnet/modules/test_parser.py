import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--training:some-name', type=str, default=None)

args = vars(parser.parse_args())
print('Finished')