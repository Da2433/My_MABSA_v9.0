import os
import csv
import pickle
import numpy as np
import torch

from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from MultimodalBert.dataset.input_example_dataset import MMInputExample, MMInputFeatures

def _read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding='utf-8') as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            lines.append(line)
        return lines

def _create_examples(lines, img_region_path, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in tqdm(enumerate(lines), desc='Loading dataset...'):
        if i == 0:
            continue
        guid = "%s-%s" % (set_type, i)
        text_a = line[3].lower()
        text_b = line[4].lower()
        label = line[1]
        img_id = line[2]

        # img_region_feat = np.load(
        #      os.path.join(img_region_path, '_att', img_id[:-4] + '.npz'))['feat']
        #  vision_feature
        # try:
        img_region_feat = np.load(
            os.path.join(img_region_path, img_id[:-4] + '.npy'))
        # except Exception as e:
        #     pass
        #     print(img_id)
        # img_region_box = np.load(
        #     os.path.join(img_region_path, '_box', img_id[:-4] + '.npy'))

        examples.append(
            MMInputExample(guid=guid, text_a=text_a, text_b=text_b, img_id=img_id, img_region_feat=img_region_feat, label=label))
    return examples



def get_data_examples(args, flag = 'train'):
    if flag == 'train':
        data_path = os.path.join(args.data_dir, args.train_data_path)
        img_region_path = os.path.join(args.image_region_dir, 'train')
    elif flag == 'valid':
        data_path = os.path.join(args.data_dir, args.valid_data_path)
        img_region_path = os.path.join(args.image_region_dir, 'dev')
    elif flag == 'test':
        data_path = os.path.join(args.data_dir, args.test_data_path)
        img_region_path = os.path.join(args.image_region_dir, 'test')
    else:
        raise ValueError('Wrong flag!!!')

    datas = _read_tsv(data_path)
    examples = _create_examples(datas, img_region_path, flag)
    # labels = ["0", "1", "2"]
    return examples


def convert_examples_to_features(examples, max_seq_length, tokenizer, logger):
    """Loads a data file into a list of `InputBatch`s."""
    label_list = ["0", "1", "2"]
    label_map = {label : i for i, label in enumerate(label_list)}

    all_input_ids, all_input_mask, all_segment_ids, all_labels = [], [], [], []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]


        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]
        if ex_index < 1:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        all_input_ids.append(input_ids)
        all_input_mask.append(input_mask)
        all_segment_ids.append(segment_ids)
        all_labels.append(label_id)

    return all_input_ids, all_input_mask, all_segment_ids, all_labels


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_mm_examples_to_features(examples, args, tokenizer, logger, flag = 'train'):
    """Loads a data file into a list of `InputBatch`s."""
    if not os.path.exists(args.processed_feature_path):
        os.makedirs(args.processed_feature_path)
    if flag == 'train':
        processed_feature_dir = os.path.join(args.processed_feature_path, 'train_cache.pkl')
    elif flag == 'valid':
        processed_feature_dir = os.path.join(args.processed_feature_path, 'valid_cache.pkl')
    elif flag == 'test':
        processed_feature_dir = os.path.join(args.processed_feature_path, 'test_cache.pkl')
    else:
        raise KeyError('Unknown flag type...')


    if os.path.exists(processed_feature_dir):
        print('[INFO] OK,pkl exist,Loading the processed datas')
        assert '.pkl' in processed_feature_dir
        with open(processed_feature_dir, 'rb') as f:
            features = pickle.load(f)
        return features
    else:
        assert not os.path.exists(processed_feature_dir)
        features = _prepare_region_bert_datas(examples, tokenizer, args, logger)

        # with open(processed_feature_dir, 'wb') as fw:
        #     pickle.dump(features, fw)
        return features

def _prepare_region_bert_datas(examples, tokenizer, args, logger):
    max_seq_length = args.max_seq_length
    max_entity_length = args.max_entity_length
    path_img = args.image_path
    crop_size = args.crop_size

    label_list = ["0", "1", "2"]
    count = 0
    label_map = {label : i for i, label in enumerate(label_list)}


    transform = transforms.Compose([
        transforms.RandomCrop(crop_size),  # args.crop_size, by default it is set to be 224
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    features = []
    sent_length_a, entity_length_b, total_length = 0, 0, 0
    for (ex_index, example) in tqdm(enumerate(examples), desc='Converting dataset to feature...'):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = tokenizer.tokenize(example.text_b)
        if len(tokens_b) >= entity_length_b:
            entity_length_b = len(tokens_b)
        if len(tokens_a) >= sent_length_a:
            sent_length_a = len(tokens_a)

        if len(tokens_b) > max_entity_length - 2:
            s2_tokens = tokens_b[:(max_entity_length - 2)]
        else:
            s2_tokens = tokens_b
        s2_tokens = ["[CLS]"] + s2_tokens + ["[SEP]"]
        s2_input_ids = tokenizer.convert_tokens_to_ids(s2_tokens)
        s2_input_mask = [1] * len(s2_input_ids)
        s2_segment_ids = [0] * len(s2_input_ids)

        # Zero-pad up to the sequence length.
        s2_padding = [0] * (max_entity_length - len(s2_input_ids))
        s2_input_ids += s2_padding
        s2_input_mask += s2_padding
        s2_segment_ids += s2_padding

        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        # if tokens_b:
        #     tokens += tokens_b + ["[SEP]"]
        #     segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        if len(tokens) >= total_length:
            total_length = len(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)
        added_input_mask = [1] * (len(input_ids) + 49)  # 1 or 49 is for encoding regional image representations

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        added_input_mask += padding
        segment_ids += padding


        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]

        # todo 不需要resnet的图像特征
        # image_name = example.img_id
        # image_path = os.path.join(path_img, image_name)
        #
        # if not os.path.exists(image_path):
        #     print(image_path)
        # try:
        #     image = image_process(image_path, transform)
        # except:
        #     count += 1
        #     # print('image has problem!')
        #     image_path_fail = os.path.join(path_img, '17_06_4705.jpg')
        #     image = image_process(image_path_fail, transform)
        image = None

        region_feat = example.img_region_feat

        # if ex_index < 1:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("tokens: %s" % " ".join(
        #         [str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info(
        #         "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #     logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            MMInputFeatures(input_ids=input_ids,
                            input_mask=input_mask,
                            added_input_mask=added_input_mask,
                            segment_ids=segment_ids,
                            s2_input_ids=s2_input_ids,
                            s2_input_mask=s2_input_mask,
                            s2_segment_ids=s2_segment_ids,
                            img_feat=image,
                            img_region_feat = region_feat,
                            label_id=label_id
                            ))

    return features

def _prepare_multimodal_datas(examples, tokenizer, args, logger):
    max_seq_length = args.max_seq_length
    max_entity_length = args.max_entity_length
    path_img = args.image_path
    crop_size = args.crop_size

    label_list = ["0", "1", "2"]
    count = 0
    label_map = {label : i for i, label in enumerate(label_list)}


    transform = transforms.Compose([
        transforms.RandomCrop(crop_size),  # args.crop_size, by default it is set to be 224
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    features = []
    sent_length_a, entity_length_b, total_length = 0, 0, 0
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = tokenizer.tokenize(example.text_b)
        if len(tokens_b) >= entity_length_b:
            entity_length_b = len(tokens_b)
        if len(tokens_a) >= sent_length_a:
            sent_length_a = len(tokens_a)

        if len(tokens_b) > max_entity_length - 2:
            s2_tokens = tokens_b[:(max_entity_length - 2)]
        else:
            s2_tokens = tokens_b
        s2_tokens = ["[CLS]"] + s2_tokens + ["[SEP]"]
        s2_input_ids = tokenizer.convert_tokens_to_ids(s2_tokens)
        s2_input_mask = [1] * len(s2_input_ids)
        s2_segment_ids = [0] * len(s2_input_ids)

        # Zero-pad up to the sequence length.
        s2_padding = [0] * (max_entity_length - len(s2_input_ids))
        s2_input_ids += s2_padding
        s2_input_mask += s2_padding
        s2_segment_ids += s2_padding

        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        if len(tokens) >= total_length:
            total_length = len(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)
        added_input_mask = [1] * (len(input_ids) + 49)  # 1 or 49 is for encoding regional image representations

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        added_input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]

        image_name = example.img_id
        image_path = os.path.join(path_img, image_name)

        if not os.path.exists(image_path):
            print(image_path)
        try:
            image = image_process(image_path, transform)
        except:
            count += 1
            # print('image has problem!')
            image_path_fail = os.path.join(path_img, '17_06_4705.jpg')
            image = image_process(image_path_fail, transform)

        if ex_index < 1:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            MMInputFeatures(input_ids=input_ids,
                            input_mask=input_mask,
                            added_input_mask=added_input_mask,
                            segment_ids=segment_ids,
                            s2_input_ids=s2_input_ids,
                            s2_input_mask=s2_input_mask,
                            s2_segment_ids=s2_segment_ids,
                            img_feat=image,
                            label_id=label_id))

    return features




def image_process(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image