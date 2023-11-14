from torch.utils.data.dataset import Dataset

from MultimodalBert.dataset.multimodal_dataset import MultimodalDataset
from MultimodalBert.dataset.multimodal_data_loader import MultimodalDataLoader

# from my_bert.tokenization import BertTokenizer
from transformers import BertTokenizer

class MultimodalDataModule(object):
    def __init__(self, args, logger):
        self.args = args
        self.num_train_steps = -1
        self.datasets = {}
        self.dataset_to_iter = {}
        self.logger = logger
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=args.pretrain_model_name,
                                                       do_lower_case=args.do_lower_case)


    def dataset(self, flag = 'train'):
        return self.datasets.get(flag, None)

    def load_dataset(self, flag = 'train'):
        self.datasets[flag] = MultimodalDataset(self.args, self.tokenizer,
                                          self.logger, flag)
        if flag == 'train':
            self.num_train_steps = int(len(self.datasets[flag].features)
                                       /self.args.batch_size
                                       / self.args.gradient_accumulation_steps
                                       * self.args.num_epoch)
    def get_dataloader(self, dataset):
        if dataset in self.dataset_to_iter:
            return self.dataset_to_iter[dataset]
        else:
            assert isinstance(dataset, MultimodalDataset)
            data_loader = MultimodalDataLoader(dataset=dataset,
                                               batch_size=self.args.batch_size,
                                               collate_fn=dataset.collate_fn,
                                               flag=dataset.flag).get_dataloader()
            self.dataset_to_iter[dataset] = data_loader
            return data_loader
