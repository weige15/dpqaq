import torch
from functools import partial


def _move_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


# Forward wrapper function to record hidden states as well as miscellaneous inputs
def decoder_layer_forward_wrap(self, orig_fn, xarr, misc_inputs, hidden_states, *args, **kwargs):
    xarr.append(hidden_states.clone().cpu())
    if len(misc_inputs) == 0:
        misc_inputs.extend([tuple(args), dict(kwargs)])
    return orig_fn(hidden_states, *args, **kwargs)

@torch.no_grad()
def getLayer0Inputs(model, dataset, device="cuda") -> tuple[list, tuple]:
    """
    Record inputs for decoder layer 0.

    Args:
        model: original model used for retrieving inputs
        dataset: dataset, probably a DataLoader
        device: device used for forward capture

    Returns:
        tuple of (xarr, misc_inputs) where
            xarr: list of recorded hidden states
            misc_inputs: tuple of miscellaneous inputs used during forward
    """
    # Hook for recording inputs
    xarr = []
    misc_inputs = []

    # Wrap forward
    curr_layer = model.model.layers[0]
    orig_forward = curr_layer.forward
    curr_layer.forward = partial(decoder_layer_forward_wrap, curr_layer, orig_forward, xarr, misc_inputs)

    # Compute upto only current layer
    orig_layers = model.model.layers
    model.model.layers = orig_layers[:1]

    model = model.to(device)

    # Run dataset
    for batch in dataset:
        b = batch['input_ids'].to(model.device)
        model(b)

    # Clean up
    curr_layer.forward = orig_forward
    model.model.layers = orig_layers
    model = model.cpu()

    return xarr, tuple(misc_inputs)


@torch.no_grad()
def getX(model, layer, module_list, layerX, misc_inputs, device="cuda") -> tuple[dict, list]:
    """
    For given decoder layer, record inputs for every linear within the layer.
    Also, get the output of the layer.

    Args:
        model: original model used for retrieving inputs
        layer: the number of current decoder layer
        module_list: list of tuples of (parent_module, name) for the linear layers within the layer
        layerX: the inputs to the decoder layer
        misc_inputs: tuple of miscellaneous inputs used during forward

    Returns:
        tuple of (xarr_d, layer_out_arr) where
            xarr_d: dictionary of recorded inputs for each linear
            layer_out_arr: list of outputs of the current layer
    """
    # Get layer
    curr_layer = model.model.layers[layer]
    curr_layer = curr_layer.to(device)

    # Hook for recording inputs
    def pre_hook(module, input):
        module.xarr.append(input[0].clone().cpu())

    # Register hook
    handle_list = []
    for parent, name in module_list:
        curr_layer._modules[parent]._modules[name].xarr = []
        handle = curr_layer._modules[parent]._modules[name].register_forward_pre_hook(pre_hook)
        handle_list.append(handle)

    # Do forward for layer
    layer_out_arr = []
    for batch in layerX:
        if len(misc_inputs) == 2 and isinstance(misc_inputs[0], tuple) and isinstance(misc_inputs[1], dict):
            replay_args = _move_to_device(misc_inputs[0], device)
            replay_kwargs = _move_to_device(misc_inputs[1], device)
            out = curr_layer(batch.to(device), *replay_args, **replay_kwargs)
        else:
            replay_args = _move_to_device(misc_inputs, device)
            out = curr_layer(batch.to(device), *replay_args)
        hidden_out = out[0] if isinstance(out, (tuple, list)) else out
        layer_out_arr.append(hidden_out.clone().cpu())

    # Clean up
    for handle in handle_list:
        handle.remove()
    curr_layer = curr_layer.cpu()

    # Harvest xarr
    xarr_d = {}
    for parent, name in module_list:
        xarr_d[name] = curr_layer._modules[parent]._modules[name].xarr

    return xarr_d, layer_out_arr

# Clean up recorded inputs
@torch.no_grad()
def clearX(model, layer, module_list):
    curr_layer = model.model.layers[layer]

    for parent, name in module_list:
        curr_layer._modules[parent]._modules[name].xarr.clear()
