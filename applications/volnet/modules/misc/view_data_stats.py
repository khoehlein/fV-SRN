import numpy as np
from matplotlib import pyplot as plt

from volnet.modules.datasets.cvol_reader import VolumetricScene

scene = VolumetricScene.from_cvol('/mnt/hdd10tb/Datasets/1k_member_ensemble_201606110000/converted_normal_anomaly/tk/member0001/t03.cvol')
feature = scene.get_active_feature()

data = feature.data[0].cpu().numpy()
print('[INFO] Data shape:', data.shape)

slices = (np.linspace(0, 1, 10) * (data.shape[0] -1)).astype(int)

fig, ax = plt.subplots(len(slices), 1, dpi=300, figsize=(10, 2 * len(slices)))

for i,s in enumerate(slices):
    current_slice = data[:, int(s), :]
    print(f'{i}: {current_slice.shape}')
    ax[i].pcolor(current_slice.T, vmin=-1.75, vmax=1.75)
plt.tight_layout()
plt.show()
plt.close()

plt.figure()
plt.hist(data.ravel(),bins=100)
plt.show()
