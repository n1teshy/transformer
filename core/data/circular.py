from core.data.seq_to_seq import SeqToSeqDataset


class CircularBatchGenerator:
    def __init__(self, dataset: SeqToSeqDataset):
        self.dataset = dataset

    def __iter__(self):
        while True:
            yield from self.dataset.batch_generator()
