from torch.utils.data import DataLoader


class CircularDataloader:
    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader

    def __iter__(self):
        while True:
            for batch in self.dataloader:
                yield batch
