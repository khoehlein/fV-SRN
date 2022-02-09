import argparse

from volnet.input_data import TrainingInputData


def build_parser():
    parser = argparse.ArgumentParser()
    TrainingInputData.init_parser(parser)
    return parser


def main():
    parser = build_parser()
    opt = vars(parser.parse_args())
    print('Finished')


if __name__ == '__main__':
    main()
