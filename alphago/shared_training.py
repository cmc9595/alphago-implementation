"""Multi-task supervision with disjoint human-policy and generated-value labels."""
import torch
import torch.nn.functional as F


def supervised_losses(outputs, human_moves, generated_values, generated_ownership,
                      ownership_weight=.5):
    policy, value, own = outputs
    h = len(human_moves)
    if len(policy) != h + len(generated_values):
        raise ValueError('Mixed batch must contain human rows followed by generated rows')
    p_loss = F.cross_entropy(policy[:h], human_moves)
    known = generated_values != 0
    v_loss = F.mse_loss(value[h:][known], generated_values[known]) if known.any() else value[h:].sum() * 0
    o_loss = F.mse_loss(own[h:], generated_ownership)
    return p_loss + v_loss + ownership_weight * o_loss, {
        'policy_ce': p_loss, 'value_mse': v_loss, 'ownership_mse': o_loss,
        'policy_acc': (policy[:h].argmax(1) == human_moves).float().mean()}


def training_mode(net, heads_only=False):
    """Warm up the new task heads; freeze pretrained BatchNorm running stats."""
    for name, parameter in net.named_parameters():
        parameter.requires_grad_(not heads_only or name.startswith(('value_', 'ownership_')))
    net.train()
    for module in net.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()  # Affine parameters still learn once their trunk/head is enabled.
