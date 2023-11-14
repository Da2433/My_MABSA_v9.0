import torch
from torch.utils.data.dataset import Dataset
from utilss.data_utils import convert_examples_to_features, \
                              get_data_examples, \
                              convert_mm_examples_to_features



class MultimodalDataset(Dataset):
    def __init__(self, args, tokenizer, logger, flag = 'train'):
        self.args = args
        self.flag = flag
        self.tokenizer = tokenizer
        self.data_examples = get_data_examples(args, flag)

        self.features = convert_mm_examples_to_features(self.data_examples,
                                                        args, tokenizer, logger)


    def __len__(self):
        return len(self.features)


    def __getitem__(self, item):
        tmp_input_ids = self.features[item].input_ids
        tmp_input_mask = self.features[item].input_mask
        tmp_added_input_mask = self.features[item].added_input_mask
        tmp_segment_ids = self.features[item].segment_ids
        tmp_s2_input_ids = self.features[item].s2_input_ids
        tmp_s2_input_mask = self.features[item].s2_input_mask
        tmp_s2_segment_ids = self.features[item].s2_segment_ids
        tmp_img_feat = self.features[item].img_feat
        tmp_img_region_feat = self.features[item].img_region_feat
        tmp_label_ids = self.features[item].label_id


        tmp = {
            'input_ids': tmp_input_ids,
            'input_mask': tmp_input_mask,
            'added_input_mask': tmp_added_input_mask,
            'segment_ids': tmp_segment_ids,
            's2_input_ids': tmp_s2_input_ids,
            's2_input_mask': tmp_s2_input_mask,
            's2_segment_ids': tmp_s2_segment_ids,
            'img_feat': tmp_img_feat,
            'img_region_feat':tmp_img_region_feat,
            'label_ids': tmp_label_ids
        }

        return tmp


    def collate_fn(self, samples):
        batch_input_ids = [s['input_ids'] for s in samples]
        batch_input_mask = [s['input_mask'] for s in samples]
        batch_added_input_mask = [s['added_input_mask'] for s in samples]
        batch_segment_ids = [s['segment_ids'] for s in samples]
        batch_s2_input_ids = [s['s2_input_ids'] for s in samples]
        batch_s2_input_mask = [s['s2_input_mask'] for s in samples]
        batch_s2_segment_ids = [s['s2_segment_ids'] for s in samples]
        # batch_img_feat = [s['img_feat'] for s in samples] #todo 不要resnet的图像特征
        batch_img_region_feat = [s['img_region_feat'] for s in samples]
        batch_label_ids = [s['label_ids'] for s in samples]



        batch_input_ids = torch.tensor(batch_input_ids, dtype=torch.long)
        batch_input_mask = torch.tensor(batch_input_mask, dtype=torch.long)
        batch_added_input_mask = torch.tensor(batch_added_input_mask, dtype=torch.long)
        batch_segment_ids = torch.tensor(batch_segment_ids, dtype=torch.long)

        batch_s2_input_ids = torch.tensor(batch_s2_input_ids, dtype=torch.long)
        batch_s2_input_mask = torch.tensor(batch_s2_input_mask, dtype=torch.long)
        batch_s2_segment_ids = torch.tensor(batch_s2_segment_ids, dtype=torch.long)
        # batch_img_feat = torch.stack(batch_img_feat, dim=0) #todo 不要resnet的图像特征
        batch_img_feat = None
        batch_img_region_feat = torch.stack([torch.from_numpy(i) for i in batch_img_region_feat], dim=0)
        batch_label_ids = torch.tensor(batch_label_ids, dtype=torch.long)



        batch_datas = {
            'input_ids': batch_input_ids,
            'input_mask': batch_input_mask,
            'added_input_mask': batch_added_input_mask,
            'segment_ids': batch_segment_ids,
            's2_input_ids': batch_s2_input_ids,
            's2_input_mask': batch_s2_input_mask,
            's2_segment_ids': batch_s2_segment_ids,
            'img_feat': batch_img_feat,
            'img_region_feat':batch_img_region_feat,
            'label_ids': batch_label_ids
        }
        return batch_datas