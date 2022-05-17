from torch.utils.data import Dataset


class SkipData(Dataset):

    def __init__(self, data: Dataset, step: int):
        super(SkipData, self).__init__()
        self.data = data
        self.step = step

    def __getitem__(self, item):
        return self.data[item * self.step]

    def __len__(self):
        return len(self.data) // self.step
