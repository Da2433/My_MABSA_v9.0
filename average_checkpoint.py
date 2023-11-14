import re
import os
import torch
import argparse
import collections


def average_checkpoints(input_files):
    param_dict = collections.OrderedDict()
    params_keys = None
    new_state = None
    num_models = len(input_files)

    for f in input_files:
        state = torch.load(f, map_location='cpu')

        if new_state is None:
            new_state = state

        model_params = state.state_dict()

        model_params_keys = list(model_params.keys())
        if params_keys is None:
            params_keys = model_params_keys
        elif params_keys != model_params_keys:
            raise KeyError(
                'For checkpoint {}, expected list of params: {}, '
                'but found: {}'.format(f, params_keys, model_params_keys)
            )

        for k in params_keys:
            p = model_params[k]
            if isinstance(p, torch.HalfTensor):
                p = p.float()
            if k not in param_dict:
                param_dict[k] = p.clone()
                # NOTE: clone() is needed in case of p is a shared parameter
            else:
                param_dict[k] += p

    avg_params = collections.OrderedDict()
    for k, v in param_dict.items():
        avg_params[k] = v
        avg_params[k] = (avg_params[k] / num_models).type_as(param_dict[k])
        # avg_params[k].div_(num_models)
    new_state = avg_params
    return new_state





def last_n_checkpoint(input_ckpt_path, num):
    pt_regexp = re.compile(r'model_(\d+)\.pt')

    files = os.listdir(input_ckpt_path)

    entries = []
    for f in files:
        m = pt_regexp.fullmatch(f)
        if m is not None:
            sort_key = int(m.group(1))
            entries.append((sort_key, m.group(0)))

    if len(entries) < num:
        raise Exception('Found {} checkpoint files but need at least {}', len(entries), num)
    return [os.path.join(input_ckpt_path, x[1]) for x in sorted(entries, reverse=True)[:num]]



def main(num_last_ckpt=None):
    parser = argparse.ArgumentParser(description='Average the params of input ckpt')

    parser.add_argument('--input_ckpt_path', default='outs/', type=str)
    parser.add_argument('--output_ckpt', default='outs/avg_model.pt', type=str)
    parser.add_argument('--num_epoch_checkpoints', default=2, type=int)

    args = parser.parse_args()

    if num_last_ckpt is not None:
        args.num_epoch_checkpoints = num_last_ckpt
    # print(args)


    assert args.num_epoch_checkpoints is not None
    input_files = last_n_checkpoint(args.input_ckpt_path, args.num_epoch_checkpoints)
    # input_files = ['outs/model_36.pt', 'outs/model_35.pt']
    input_files.append('outs/model_b.pt')

    print('averaging checkpoints: ', input_files)

    avg_state = average_checkpoints(input_files)

    torch.save(avg_state, args.output_ckpt)
    print('Finished writing averaged checkpoint to {}.'.format(args.output_ckpt))



if __name__ == '__main__':
    main()


    import logging
    from utilss.args_utils import get_args, update_args
    from MultimodalBert.trainer.multimodal_trainer import MultimodalTrainer
    import warnings

    warnings.filterwarnings('ignore')

    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)

    args = get_args()
    args = update_args(args)
    my_trainer = MultimodalTrainer(args, logger)
    # my_trainer.do_train()
    # my_trainer.do_test()
    my_trainer.do_test_with_avg_model()