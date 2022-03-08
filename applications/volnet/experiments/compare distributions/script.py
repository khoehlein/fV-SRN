import matplotlib.pyplot as plt
import numpy as np

from volnet.modules.datasets.cvol_reader import VolumetricScene

print('[INFO] Loading data')
ejecta_file = '/home/hoehlein/PycharmProjects/deployment/delllat94/fvsrn/applications/volumes/Ejecta/snapshot_070_256.cvol'
scene_ejecta = VolumetricScene.from_cvol(ejecta_file)
data_ejecta = scene_ejecta.get_active_feature().data.cpu().numpy()
member_file = '/mnt/hdd10tb/Datasets/1k_member_ensemble_201606110000/converted/tk/member0003/t04.cvol'
scene_member = VolumetricScene.from_cvol(member_file)
data_member = scene_member.get_active_feature().data.cpu().numpy()

print('[INFO] Plotting...')
NUM_BINS = 100
fig, ax = plt.subplots(2, 1, figsize=(15, 10), dpi=300)
ax[0].hist(data_ejecta.ravel(), bins=np.arange(NUM_BINS) / NUM_BINS)
ax[1].hist(data_member.ravel(), bins=np.arange(NUM_BINS) / NUM_BINS)
plt.tight_layout()
plt.show()
plt.close()

print('[INFO] Finished')
