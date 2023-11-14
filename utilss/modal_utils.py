import os
import torch

def save_model(model, model_save_path, model_name='model'):
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    torch.save(model, os.path.join(model_save_path, model_name + '.pt'))


def load_model(args, model_name:str = 'model_b'):
    model_path = os.path.join(args.ckpt_out_file, model_name + '.pt')

    model = torch.load(model_path, map_location='cpu')
    return model


def load_avg_model_state(args):
    model_path = os.path.join(args.ckpt_out_file, 'avg_model.pt')

    model_state = torch.load(model_path)
    return model_state


