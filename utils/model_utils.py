import torch

import pdb

# Saves the model along with the args for the model
def save_model(model, args, save_dir, model_name, should_print=True):
    save_path = '%s/model_%s' % (save_dir, str(model_name))

    save_dict = {'model': model.state_dict(), 'args': args}
    torch.save(save_dict, save_path)

    if should_print:
        print('Model saved to: %s' % save_path)


# Load the model along with args for the model
def load_model(path, model_class, device):
    model_dict = torch.load(path)
    args = model_dict['args']

    if device is None:
        device = args.device

    model = model_class(args)
    model.load_state_dict(model_dict['model'])
    model.to(device)
    return model, args


def get_num_params(model):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params
