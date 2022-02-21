import torch.nn as nn

from modules.token_embedders.pretrained_encoder import PretrainedEncoder
from utils.nn_utils import batched_index_select, gelu


class PretrainedEmbedModel(nn.Module):
    """An embeddding layer with pre-trained model
    """
    def __init__(self, args, vocab):
        """Constructs `PretrainedEmbedModel` components and
        sets `PretrainedEmbedModel` parameters

        Arguments:
            args {dict} -- config parameters for constructing multiple models
            vocab {Vocabulary} -- vocabulary
        """

        super().__init__()
        self.activation = gelu
        self.pretrained_encoder = PretrainedEncoder(pretrained_model_name=args.pretrained_model_name,
                                                    trainable=args.fine_tune,
                                                    output_size=args.ptm_output_size,
                                                    activation=self.activation,
                                                    dropout=args.ptm_dropout)
        self.encoder_output_size = self.pretrained_encoder.get_output_dims()

    def forward(self, batch_inputs):
        """Propagates forwardly

        Arguments:
            batch_inputs {dict} -- batch input data
        """

        if 'subword_segment_ids' in batch_inputs:
            batch_seq_pretrained_encoder_repr, batch_cls_repr, batch_pretrained_hidden_states, batch_pretrained_attentions = self.pretrained_encoder(
                batch_inputs['subword_tokens'], batch_inputs['subword_segment_ids'])
            batch_inputs['batch_pretrained_attentions'] = batch_pretrained_attentions
        else:
            batch_seq_pretrained_encoder_repr, batch_cls_repr, batch_pretrained_hidden_states, batch_pretrained_attentions = self.pretrained_encoder(
                batch_inputs['subword_tokens'])

        batch_seq_tokens_encoder_repr = batched_index_select(batch_seq_pretrained_encoder_repr,
                                                             batch_inputs['subword_tokens_index'])

        batch_inputs['seq_encoder_reprs'] = batch_seq_tokens_encoder_repr
        batch_inputs['seq_cls_repr'] = batch_cls_repr

    def get_hidden_size(self):
        """Returns embedding dimensions

        Returns:
            int -- embedding dimensions
        """

        return self.encoder_output_size
