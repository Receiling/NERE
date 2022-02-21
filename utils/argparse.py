import os
import logging

import configargparse

from utils.logging_utils import init_logger
from utils.parse_action import StoreLoggingLevelAction


class ConfigurationParer():
    """Customized configuration parser.
    """
    def __init__(self,
                 config_file_parser_class=configargparse.YAMLConfigFileParser,
                 formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
                 **kwargs):
        """Decides configuration parser and formatter.

        Keyword Arguments:
            config_file_parser_class {configargparse.ConfigFileParser} -- configuration file parser (default: {configargparse.YAMLConfigFileParser})
            formatter_class {configargparse.ArgumentDefaultsHelpFormatter} -- configuration formatter (default: {configargparse.ArgumentDefaultsHelpFormatter})
        """

        self.parser = configargparse.ArgumentParser(config_file_parser_class=config_file_parser_class,
                                                    formatter_class=formatter_class,
                                                    **kwargs)

        self.parser.add('-config_file',
                        '--config_file',
                        required=False,
                        is_config_file_arg=True,
                        help='config file path')
        self.parser.add('-save_dir', '--save_dir', type=str, required=True, help='directory for saving.')

    def add_input_args(self):
        """Adds input arguments: dataset, pre-trained models, etc.
        """

        # dataset arguments
        group = self.parser.add_argument_group('Dataset')
        group.add('-data_dir', '--data_dir', type=str, required=True, help='dataset directory.')
        group.add('-train_file', '--train_file', type=str, required=False, help='train data file.')
        group.add('-dev_file', '--dev_file', type=str, required=False, help='dev data file.')
        group.add('-test_file', '--test_file', type=str, required=False, help='test data file.')
        group.add('-max_sent_len', '--max_sent_len', type=int, default=200, help='max sentence length.')
        group.add('-max_subword_len', '--max_subword_len', type=int, default=512, help='max sentence length.')
        group.add('-entity_schema', '--entity_schema', type=str, required=False, help='entity tag schema.')
        group.add('-low_case', '--low_case', type=int, required=False, help='tansform to low case')

        # pre-trained model arguments
        group = self.parser.add_argument_group('Pretrained-Model')
        group.add('-pretrained_embeddings_file',
                  '--pretrained_embeddings_file',
                  type=str,
                  required=False,
                  help='pretrained word embeddings file.')

    def add_model_args(self):
        """Adds model (network) arguments: embedding size, hidden size, etc.
        """

        # embedding arguments
        group = self.parser.add_argument_group('Embeddings')
        group.add('-embedding_dims', '--embedding_dims', type=int, required=False, help='word embedding dimensions.')
        group.add('-word_dims', '--word_dims', type=int, required=False, help='word embedding dimensions.')
        group.add('-char_dims', '--char_dims', type=int, required=False, help='char embedding dimensions.')
        group.add('-char_batch_size', '--char_batch_size', type=int, required=False, help='char embedding batch size.')
        group.add('-char_kernel_sizes',
                  '--char_kernel_sizes',
                  default=[2, 3],
                  nargs='*',
                  type=int,
                  help='char embedding convolution kernel size list.')
        group.add('-char_output_channels',
                  '--char_output_channels',
                  type=int,
                  default=25,
                  help='char embedding output dimensions.')
        group.add('-embedding_dropout',
                  '--embedding_dropout',
                  type=float,
                  default=0.0,
                  help='embedding module dropout rate.')

        # entity model arguments
        group = self.parser.add_argument_group('Entity-Model')
        group.add('-embedding_model',
                  '--embedding_model',
                  type=str,
                  choices=["word_char", "pretrained"],
                  default="pretrained",
                  help='embedding model.')
        group.add('-entity_model',
                  '--entity_model',
                  type=str,
                  choices=["joint", "pipeline"],
                  default="pipeline",
                  help='entity recognition model.')
        group.add('-lstm_layers', '--lstm_layers', type=int, default=1, help='the number of lstm layers.')
        group.add('-lstm_hidden_unit_dims',
                  '--lstm_hidden_unit_dims',
                  type=int,
                  default=128,
                  help='lstm hidden unit dimensions.')
        group.add('-lstm_dropout', '--lstm_dropout', type=float, default=0.0, help='the dropout of the lstm layer.')
        group.add('-entity_cnn_kernel_sizes',
                  '--entity_cnn_kernel_sizes',
                  default=[2, 3],
                  nargs='*',
                  type=int,
                  help='entity span cnn convolution kernel size list.')
        group.add('-entity_cnn_output_channels',
                  '--entity_cnn_output_channels',
                  type=int,
                  default=25,
                  help='entity span cnn convolution output dimensions.')
        group.add('-ent_output_size', '--ent_output_size', type=int, default=0, help='entity encoder output size.')
        group.add('-span_batch_size', '--span_batch_size', type=int, default=-1, help='cache span feature batch size.')
        group.add('-ent_batch_size', '--ent_batch_size', type=int, default=-1, help='ent convolution batch size.')

        # relation model arguments
        group = self.parser.add_argument_group('Relation-Model')
        group.add('-schedule_k', '--schedule_k', type=float, default='1.0', help='scheduled sampling rate.')
        group.add('-context_cnn_kernel_sizes',
                  '--context_cnn_kernel_sizes',
                  default=[2, 3],
                  nargs='*',
                  type=int,
                  help='entity span cnn convolution kernel size list.')
        group.add('-context_cnn_output_channels',
                  '--context_cnn_output_channels',
                  type=int,
                  default=25,
                  help='context span cnn convolution output dimensions.')
        group.add('-context_output_size',
                  '--context_output_size',
                  type=int,
                  default=0,
                  help='context encoder output size.')
        group.add('-ent_mention_output_size',
                  '--ent_mention_output_size',
                  type=int,
                  default=0,
                  help='entity mention model output size.')
        group.add('-dropout', '--dropout', type=float, default=0.5, help='dropout rate.')

        # pre-trained model arguments
        group = self.parser.add_argument_group('Pre-trained')
        group.add('-pretrained_model_name',
                  '--pretrained_model_name',
                  type=str,
                  required=False,
                  help='pre-trained model name.')
        group.add('-ptm_output_size', '--ptm_output_size', type=int, default=768, help='pre-trained model output size.')
        group.add('-ptm_dropout', '--ptm_dropout', type=float, default=0.1, help='pre-trained model dropout rate.')
        group.add('--fine_tune', '--fine_tune', action='store_true', help='fine-tune pretrained model.')

    def add_optimizer_args(self):
        """Adds optimizer arguments
        """

        # Adam arguments
        group = self.parser.add_argument_group('Adam')
        group.add('-adam_beta1', '--adam_beta1', type=float, default=0.9, help="The beta1 parameter used by Adam. ")
        group.add('-adam_beta2', '--adam_beta2', type=float, default=0.999, help="The beta2 parameter used by Adam. ")
        group.add('-adam_epsilon', '--adam_epsilon', type=float, default=1e-6, help='adam epsilon')
        group.add('-adam_weight_decay_rate',
                  '--adam_weight_decay_rate',
                  type=float,
                  default=0.0,
                  help='adam weight decay rate.')

    def add_run_args(self):
        """Adds running arguments: learning rate, batch size, etc.
        """

        # training arguments
        group = self.parser.add_argument_group('Training')
        group.add('-seed', '--seed', type=int, default=5216, help='radom seed.')
        group.add('-epochs', '--epochs', type=int, default=1000, help='training epochs.')
        group.add('-pretrain_epochs', '--pretrain_epochs', type=int, default=0, help='pretrain epochs.')
        group.add('-warmup_rate', '--warmup_rate', type=float, default=0.0, help='warmup rate.')
        group.add('-early_stop', '--early_stop', type=int, default=50, help='early stop threshold.')
        group.add('-train_batch_size', '--train_batch_size', type=int, default=200, help='batch size during training.')
        self.parser.add('-gradient_clipping',
                        '--gradient_clipping',
                        type=float,
                        default=1.0,
                        help='gradient clipping threshold.')
        group.add('-gradient_accumulation_steps',
                  '--gradient_accumulation_steps',
                  type=int,
                  default=1,
                  help='Number of updates steps to accumulate before performing a backward/update pass.')
        group.add('-continue_training', '--continue_training', action='store_true', help='continue training from last.')
        self.parser.add('--learning_rate',
                        '-learning_rate',
                        type=float,
                        default=3e-5,
                        help="Starting learning rate. "
                        "Recommended settings: sgd = 1, adagrad = 0.1, "
                        "adadelta = 1, adam = 0.001")
        self.parser.add('--ptm_learning_rate',
                        '-ptm_learning_rate',
                        type=float,
                        default=3e-5,
                        help="learning rate for pre-trained models, should be smaller than followed parts.")

        # testing arguments
        group = self.parser.add_argument_group('Testing')
        group.add('-test', '--test', action='store_true', help='testing mode')
        group.add('-test_batch_size', '--test_batch_size', type=int, default=100, help='batch size during testing.')
        group.add('-validate_every',
                  '--validate_every',
                  type=int,
                  default=4000,
                  help='output result every n samples during validating.')

        # gpu arguments
        group = self.parser.add_argument_group('GPU')
        group.add('-device',
                  '--device',
                  type=int,
                  default=-1,
                  help='cpu: device = -1, gpu: gpu device id(device >= 0).')

        # logging arguments
        group = self.parser.add_argument_group('logging')
        group.add('-logging_steps', '--logging_steps', type=int, default=10, help='Logging every N update steps.')
        group.add('-tensorboard', '--tensorboard', action='store_true', help='turn on tensorboard.')
        group.add('-root_log_level',
                  '--root_log_level',
                  type=str,
                  action=StoreLoggingLevelAction,
                  choices=StoreLoggingLevelAction.CHOICES,
                  default="DEBUG",
                  help='root logging out level.')
        group.add('-console_log_level',
                  '--console_log_level',
                  type=str,
                  action=StoreLoggingLevelAction,
                  choices=StoreLoggingLevelAction.CHOICES,
                  default="NOTSET",
                  help='console logging output level.')
        group.add('-log_file', '--log_file', type=str, required=False, help='logging file during running.')
        group.add('-file_log_level',
                  '--file_log_level',
                  type=str,
                  action=StoreLoggingLevelAction,
                  choices=StoreLoggingLevelAction.CHOICES,
                  default="NOTSET",
                  help='file logging output level.')

    def parse_args(self):
        """Parses arguments and initializes logger

        Returns:
            dict -- config arguments
        """

        args = self.parser.parse_args()

        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        args.best_model_path = os.path.join(args.save_dir, 'best_model')
        args.last_model_path = os.path.join(args.save_dir, 'last_model')
        args.vocabulary_file = os.path.join(args.save_dir, 'vocabulary.pickle')
        args.model_checkpoints_dir = os.path.join(args.save_dir, 'model_ckpts')

        if not os.path.exists(args.model_checkpoints_dir):
            os.makedirs(args.model_checkpoints_dir)

        if args.tensorboard:
            args.tensorboard_dir = os.path.join(args.save_dir, 'tb_output')
            if not os.path.exists(args.tensorboard_dir):
                os.makedirs(args.tensorboard_dir)

        assert os.path.exists(args.data_dir), f"dataset directory {args.data_dir} not exists !!!"
        for file in ['train_file', 'dev_file', 'test_file']:
            if getattr(args, file, None) is not None:
                setattr(args, file, os.path.join(args.data_dir, getattr(args, file, None)))

        if getattr(args, 'log_file', None) is not None:
            args.log_file = os.path.join(args.save_dir, args.log_file)
            assert not os.path.exists(args.log_file), f"log file {args.log_file} exists !!!"

        init_logger(root_log_level=getattr(args, 'root_log_level', logging.DEBUG),
                    console_log_level=getattr(args, 'console_log_level', logging.NOTSET),
                    log_file=getattr(args, 'log_file', None),
                    log_file_level=getattr(args, 'log_file_level', logging.NOTSET))

        return args

    def format_values(self):
        return self.parser.format_values()
