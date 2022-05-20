import torch
import pyrenderer


def load_data(variable_name: str, norm: str, member: int):
    path = '/home/hoehlein/data/1000_member_ensemble/cvol/single_variable/{}-min-max_scaling/{}/member{:04d}/t04.cvol'.format(norm, variable_name, member)
    volume = pyrenderer.Volume(path)
    feature = volume.get_feature(0)
    level = feature.get_level(0)
    data = level.to_tensor().data.cpu().numpy()[0]
    return data


def main():
    variable_name = 'dbz'
    data = load_data(variable_name, 'local', 1)
    print(data.min(), data.max())


if __name__ == '__main__':
    main()
