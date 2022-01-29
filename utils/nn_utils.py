import functools
import logging

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import math
import numpy as np

logger = logging.getLogger(__name__)


def get_device_of(tensor):
    """Returns the device of the tensor
    refer to https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py

    Arguments:
        tensor {tensor} -- tensor

    Returns:
        int -- device
    """

    if not tensor.is_cuda:
        return -1
    else:
        return tensor.get_device()


def get_range_vector(size, device):
    """Returns a range vector with the desired size, starting at 0
    the CUDA implementation is meant to avoid copy data from CPU to GPU
    refer to https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py

    Arguments:
        size {int} -- the size of range
        device {int} -- device

    Returns:
        torch.Tensor -- range vector
    """

    if device > -1:
        return torch.cuda.LongTensor(size, device=device).fill_(1).cumsum(0) - 1
    else:
        return torch.arange(0, size, dtype=torch.long)


def flatten_and_batch_shift_indices(indices, sequence_length):
    """Returns a vector that correctly indexes into the flattened target,
    the sequence length of the target must be provided to compute the appropriate offsets.
    refer to https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py

    Arguments:
        indices {tensor} -- index tensor
        sequence_length {int} -- sequence length

    Returns:
        tensor -- offset index tensor
    """

    # Shape: (batch_size)
    if torch.max(indices) >= sequence_length or torch.min(indices) < 0:
        raise RuntimeError("All elements in indices should be in range (0, {})".format(sequence_length - 1))
    offsets = get_range_vector(indices.size(0), get_device_of(indices)) * sequence_length
    for _ in range(len(indices.size()) - 1):
        offsets = offsets.unsqueeze(1)

    # Shape: (batch_size, d_1, ..., d_n)
    offset_indices = indices + offsets

    # Shape: (batch_size * d_1 * ... * d_n)
    offset_indices = offset_indices.view(-1)
    return offset_indices


def batched_index_select(target, indices, flattened_indices=None):
    """Returns selected values in the target with respect to the provided indices,
    which have size ``(batch_size, d_1, ..., d_n, embedding_size)``
    refer to https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py

    Arguments:
        target {torch.Tensor} -- target tensor
        indices {torch.LongTensor} -- index tensor

    Keyword Arguments:
        flattened_indices {Optional[torch.LongTensor]} -- flattened index tensor (default: {None})

    Returns:
        torch.Tensor -- selected tensor
    """

    if flattened_indices is None:
        # Shape: (batch_size * d_1 * ... * d_n)
        flattened_indices = flatten_and_batch_shift_indices(indices, target.size(1))

    # Shape: (batch_size * sequence_length, embedding_size)
    flattened_target = target.view(-1, target.size(-1))

    # Shape: (batch_size * d_1 * ... * d_n, embedding_size)
    flattened_selected = flattened_target.index_select(0, flattened_indices)
    selected_shape = list(indices.size()) + [target.size(-1)]
    # Shape: (batch_size, d_1, ..., d_n, embedding_size)
    selected_targets = flattened_selected.view(*selected_shape)
    return selected_targets


def batched_2d_index_select(target, indices):
    """Return values in the target with respect to the provided
    indices in two dimensions.

    Args:
        target (torch.Tensor): target tensor, [batch_size, channel, w, h]
        indices (torch.LongTensor): indices, [batch_size, d]
    
    Returns:
        torch.Tensor -- selected tensor
    """
    _, channel, _, h = target.size()
    indices_1 = indices.unsqueeze(1).unsqueeze(-1).expand(-1, channel, -1, h)
    indices_2 = indices.unsqueeze(1).unsqueeze(1).expand(-1, channel, indices.size()[-1], -1)
    return target.gather(2, indices_1).gather(3, indices_2)


def get_padding_vector(size, dtype, device):
    """Initializes padding unit

    Arguments:
        size {int} -- padding unit size
        dtype {torch.dtype} -- dtype
        device {int} -- device = -1 if cpu, device >= 0 if gpu

    Returns:
        tensor -- padding tensor
    """

    pad = torch.zeros(size, dtype=dtype)
    if device > -1:
        pad = pad.cuda(device=device, non_blocking=True)
    return pad


def array2tensor(array, dtype, device):
    """Transforms numpy array to tensor

    Arguments:
        array {numpy.array} -- numpy array
        dtype {torch.dtype} -- torch dtype
        device {int} -- device = -1 if cpu, device >= 0 if gpu

    Returns:
        tensor -- tensor
    """
    tensor = torch.as_tensor(array, dtype=dtype)
    if device > -1:
        tensor = tensor.cuda(device=device, non_blocking=True)
    return tensor


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
        refer to: https://github.com/huggingface/pytorch-transformers/blob/master/pytorch_transformers/modeling_bert.py
    """

    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def pad_vecs(vecs, padding_size, dtype, device):
    """Pads vectors for batch

    Arguments:
        vecs {list} -- vector list
        padding_size {int} -- padding dims
        dtype {torch.dtype} -- dtype
        device {int} -- device = -1 if cpu, device >= 0 if gpu

    Returns:
        tensor -- padded vectors
    """
    max_length = max(len(vec) for vec in vecs)

    if max_length == 0:
        pad_vecs = torch.cat([get_padding_vector((1, padding_size), dtype, device).unsqueeze(0) for _ in vecs], 0)
        # mask_vecs = array2tensor(np.ones((len(vecs), 1)), vecs.dtype,
        #                          get_device_of(vecs))
        return pad_vecs  # , mask_vecs

    pad_vecs = []
    # mask_vecs = []
    for vec in vecs:
        pad_vec = torch.cat(vec + [get_padding_vector((1, padding_size), dtype, device)] * (max_length - len(vec)),
                            0).unsqueeze(0)
        # mask_vec = array2tensor(
        #     np.array([1] * len(vec) + [0] *
        #              (max_length - len(vec))).reshape(1, max_length),
        #     vecs.dtype, get_device_of(vecs))

        assert pad_vec.size() == (1, max_length, padding_size), "the size of pad vector is not correct"
        # assert mask_vec.size() == (
        #     1, max_length), "the size of mask vector is not correct"

        pad_vecs.append(pad_vec)
        # mask_vecs.append(mask_vec)
    return torch.cat(pad_vecs, 0)  # , torch.cat(mask_vecs, 0)


def get_bilstm_minus(batch_seq_encoder_repr, span_list, seq_lens):
    """TGets span representation using bilstm minus

    Arguments:
        batch_seq_encoder_repr {list} -- batch sequence encoder representation
        span_list {list} -- span list
        seq_lens {list} -- sequence length list

    Returns:
        tensor -- span representation vector
    """

    assert len(batch_seq_encoder_repr) == len(
        span_list), "the length of batch seq encoder repr is not equal to span list's length"

    assert len(span_list) == len(seq_lens), "the length of span list is not equal to batch seq lens's length"

    hidden_size = batch_seq_encoder_repr.size(-1)
    span_vecs = []
    for seq_encoder_repr, (s, e), seq_len in zip(batch_seq_encoder_repr, span_list, seq_lens):
        rnn_output = seq_encoder_repr[:seq_len]
        forward_rnn_output, backward_rnn_output = rnn_output.split(hidden_size // 2, 1)
        forward_span_vec = get_forward_segment(forward_rnn_output, s, e, get_device_of(forward_rnn_output))
        backward_span_vec = get_backward_segment(backward_rnn_output, s, e, get_device_of(backward_rnn_output))
        span_vec = torch.cat([forward_span_vec, backward_span_vec], 0).unsqueeze(0)
        span_vecs.append(span_vec)
    return torch.cat(span_vecs, 0)


def get_forward_segment(forward_rnn_output, s, e, device):
    """Gets span representation in forward rnn

    Arguments:
        forward_rnn_output {tensor} -- forward rnn output
        s {int} -- span start
        e {int} -- span end
        device {int} -- device

    Returns:
        tensor -- span representaion vector
    """

    seq_len, hidden_size = forward_rnn_output.size()
    if s >= e:
        vec = torch.zeros(hidden_size, dtype=forward_rnn_output.dtype)

        if device > -1:
            vec = vec.cuda(device=device, non_blocking=True)
        return vec

    if s == 0:
        return forward_rnn_output[e - 1]
    return forward_rnn_output[e - 1] - forward_rnn_output[s - 1]


def get_backward_segment(backward_rnn_output, s, e, device):
    """Gets span representation in backward rnn

    Arguments:
        forward_rnn_output {tensor} -- backward rnn output
        s {int} -- span start
        e {int} -- span end
        device {int} -- device

    Returns:
        tensor -- span representaion vector
    """

    seq_len, hidden_size = backward_rnn_output.size()
    if s >= e:
        vec = torch.zeros(hidden_size, dtype=backward_rnn_output.dtype)

        if device > -1:
            vec = vec.cuda(device=device, non_blocking=True)
        return vec

    if e == seq_len:
        return backward_rnn_output[s]
    return backward_rnn_output[s] - backward_rnn_output[e]


def get_dist_vecs(span_list, max_sent_len, device):
    """Gets distance embedding

    Arguments:
        span_list {list} -- span list

    Returns:
        tensor -- distance embedding vector
    """

    dist_vecs = []
    for s, e in span_list:
        assert s <= e, "span start is greater than end"

        vec = torch.Tensor(np.eye(max_sent_len)[e - s])
        if device > -1:
            vec = vec.cuda(device=device, non_blocking=True)

        dist_vecs.append(vec)

    return torch.stack(dist_vecs)


def get_conv_vecs(batch_token_repr, span_list, span_batch_size, conv_layer):
    """Gets span vector representation through convolution layer

    Arguments:
        batch_token_repr {list} -- batch token representation
        span_list {list} -- span list
        span_batch_size {int} -- span convolutuion batch size
        conv_layer {nn.Module} -- convolution layer

    Returns:
        tensor -- conv vectors
    """

    assert len(batch_token_repr) == len(span_list), "the length of batch token repr is not equal to span list's length"

    span_vecs = []
    for token_repr, (s, e) in zip(batch_token_repr, span_list):
        if s == e:
            span_vecs.append([])
            continue

        span_vecs.append(list(token_repr[s:e].split(1)))

    span_conv_vecs = []
    for id in range(0, len(span_vecs), span_batch_size):
        span_pad_vecs = pad_vecs(span_vecs[id:id + span_batch_size], conv_layer.get_input_dims(),
                                 batch_token_repr[0].dtype, get_device_of(batch_token_repr[0]))
        span_conv_vecs.append(conv_layer(span_pad_vecs))
    return torch.cat(span_conv_vecs, dim=0)


def get_exponential_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, gamma, last_epoch=-1):
    """Create a schedule with a learning rate that decreases exponentially from
    the initial lr set in the optimizer to 0, after a warmup period during which it increases linearly
    from 0 to the initial lr set in the optimizer.

    Args:
        optimizer (torch.optim.Optimizer):  The optimizer for which to schedule the learning rate.
        num_warmup_steps (int): The number of steps for the warmup phase.
        num_training_steps (int): The total number of training steps.
        gamma ([type]): Multiplicative factor of learning rate decay.
        last_epoch (int, optional): The index of the last epoch when resuming training.. Defaults to -1.
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, gamma**(num_training_steps - num_warmup_steps))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_n_trainable_parameters(model):
    """Calculates the number of trainable parameters of the model

    Arguments:
        model {nn.Module} -- model

    Returns:
        int -- the number of trainable parameters of the model
    """

    cnt = 0
    for param in list(model.parameters()):
        if param.requires_grad:
            cnt += functools.reduce(lambda x, y: x * y, list(param.size()), 1)
    return cnt


def js_div(p, q, reduction='batchmean'):
    """Calculates Jensen Shannon Divergence (JSD).

    Args:
        p (tensor): distribution p
        q (tensor): distribution q
        reduction (str, optional): reduction. Defaults to 'batchmean'.

    Returns:
        tensor: JS divergence
    """

    m = 0.5 * (p + q)
    return (F.kl_div(p, m, reduction=reduction) + F.kl_div(q, m, reduction=reduction)) * 0.5
