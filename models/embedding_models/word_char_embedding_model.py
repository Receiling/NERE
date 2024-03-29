import logging

import torch
import torch.nn as nn

from inputs.embedding_readers import glove_reader
from inputs.embedding_weight import load_embedding_weight
from modules.token_embedders.token_encoder import TokenEncoder
from modules.token_embedders.embedding import Embedding
from modules.token_embedders.char_token_encoder import CharTokenEncoder
from modules.seq2vec_encoders.cnn_encoder import CNNEncoder

logger = logging.getLogger(__name__)


class WordCharEmbedModel(nn.Module):
    """An embedding layer which consists of word embedding and cnn char embedding
    """
    def __init__(self, args, vocab):
        """Constructs `WordCharEmbedModel` components and
        sets `WordCharEmbedModel` parameters

        Arguments:
            args {dict} -- config parameters for constructing multiple models
            vocab {Vocabulary} -- vocabulary
        """

        super().__init__()
        self.vocab = vocab

        glove = glove_reader(args.pretrained_embeddings_file, args.embedding_dims)
        weight = load_embedding_weight(vocab, 'tokens', glove, args.embedding_dims)
        weight = torch.from_numpy(weight).float()
        logger.info("`tokens` size: {}, `glove` size: {}, intersetion `tokens` and `glove` of size: {}".format(
            vocab.get_vocab_size('tokens'), len(glove), len(set(glove)
                                                            & set(vocab.get_namespace_tokens('tokens')))))

        self.word_embedding = Embedding(vocab_size=self.vocab.get_vocab_size('tokens'),
                                        embedding_dim=args.embedding_dims,
                                        padding_idx=self.vocab.get_padding_index('tokens'),
                                        weight=weight,
                                        dropout=args.embedding_dropout)
        self.char_embedding = Embedding(vocab_size=self.vocab.get_vocab_size('char_tokens'),
                                        embedding_dim=args.char_dims,
                                        padding_idx=self.vocab.get_padding_index('char_tokens'),
                                        dropout=args.embedding_dropout)
        self.char_encoder = CNNEncoder(input_size=args.char_dims,
                                       num_filters=args.char_output_channels,
                                       ngram_filter_sizes=args.char_kernel_sizes,
                                       dropout=args.embedding_dropout)
        char_token_encoder = CharTokenEncoder(char_embedding=self.char_embedding, encoder=self.char_encoder)

        self.token_encoder = TokenEncoder(word_embedding=self.word_embedding,
                                          char_embedding=char_token_encoder,
                                          char_batch_size=args.char_batch_size)

        self.embedding_dims = args.word_dims + self.char_encoder.get_output_dims()

    def forward(self, batch_inputs):
        """Propagates forwardly

        Arguments:
            batch_inputs {dict} -- batch input data
        """

        batch_inputs['seq_encoder_reprs'] = self.token_encoder(batch_inputs['tokens'], batch_inputs['char_tokens'])

    def get_hidden_size(self):
        """Returns embedding dimensions
        
        Returns:
            int -- embedding dimensions
        """

        return self.embedding_dims
