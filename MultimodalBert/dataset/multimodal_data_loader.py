from torch.utils.data.dataloader import DataLoader


class MultimodalDataLoader(object):
    def __init__(self, dataset, batch_size, collate_fn, flag = 'train'):
        self.dataset = dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.shuffle = True if flag == 'train' else False

    def get_dataloader(self):
        dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            shuffle=self.shuffle
        )

        return dataloader