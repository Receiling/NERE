import os
import inspect
import logging

import configargparse
import torch

from utils.logging import logger, init_logger
from utils.parse_action import StoreLoggingLevelAction, CheckPathAction


class ConfigurationParer():
    """This class defines customized configuration parser
    """

    def __init__(self,
                 config_file_parser_class=configargparse.YAMLConfigFileParser,
                 formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
                 **kwargs):
        """This funtion decides config parser and formatter
        
        Keyword Arguments:
            config_file_parser_class {configargparse.ConfigFileParser} -- config file parser (default: {configargparse.YAMLConfigFileParser})
            formatter_class {configargparse.ArgumentDefaultsHelpFormatter} -- config formatter (default: {configargparse.ArgumentDefaultsHelpFormatter})
        """
        
        self.parser = configargparse.ArgumentParser(
            config_file_parser_class=config_file_parser_class,
            formatter_class=formatter_class,
            **kwargs)

    def add_save_cfgs(self):
        """This function adds saving path arguments: config file, model file...
        """

        # config file configurations
        group = self.parser.add_argument_group('Config-File')
        group.add('-config_file',
                  '--config_file',
                  required=False,
                  is_config_file_arg=True,
                  help='config file path')
        group.add('-save_config',
                  '--save_config',
                  required=False,
                  is_write_out_config_file_arg=True,
                  help='config file save path')

        # model file configurations
        group = self.parser.add_argument_group('Model-File')
        group.add('-save_dir',
                  '--save_dir',
                  type=str,
                  required=True,
                  help='save folder.')
        group.add('-best_model_path',
                  '--best_model_path',
                  type=str,
                  action=CheckPathAction,
                  required=True,
                  help='save best model path.')
        group.add('-last_model_path',
                  '--last_model_path',
                  type=str,
                  required=False,
                  help='load last model path.')
        group.add('-train_model_dir',
                  '--train_model_dir',
                  type=str,
                  required=True,
                  help='save training model path.')

    def add_data_cfgs(self):
        """This function adds dataset arguments: data file path...
        """

        self.parser.add('-pretrained_embeddings_file',
                        '--pretrained_embeddings_file',
                        type=str,
                        required=False,
                        help='pretrained word embeddings file.')
        self.parser.add('-vocabulary_file',
                        '--vocabulary_file',
                        type=str,
                        required=False,
                        help='vocabulary file.')
        self.parser.add('-train_file',
                        '--train_file',
                        type=str,
                        required=True,
                        help='train data file.')
        self.parser.add('-dev_file',
                        '--dev_file',
                        type=str,
                        required=True,
                        help='dev data file.')
        self.parser.add('-test_file',
                        '--test_file',
                        type=str,
                        required=False,
                        help='test data file.')
        self.parser.add('-max_sent_len',
                        '--max_sent_len',
                        type=int,
                        default=200,
                        help='max sentence length.')
        self.parser.add('-max_wordpiece_len',
                        '--max_wordpiece_len',
                        type=int,
                        default=512,
                        help='max sentence length.')
        self.parser.add('-entity_schema',
                        '--entity_schema',
                        type=str,
                        required=True,
                        help='entity tag schema.')
        self.parser.add('-low_case',
                        '--low_case',
                        type=int,
                        required=True,
                        help='tansform to low case')
        self.parser.add('-test',
                        '--test',
                        action='store_true',
                        help='testing mode')

    def add_model_cfgs(self):
        """This function adds model (network) arguments: embedding, hidden unit...
        """

        # embedding configurations
        group = self.parser.add_argument_group('Embeddings')
        group.add('-embedding_dims',
                  '--embedding_dims',
                  type=int,
                  required=False,
                  help='word embedding dimensions.')
        group.add('-word_dims',
                  '--word_dims',
                  type=int,
                  required=True,
                  help='word embedding dimensions.')
        group.add('-char_dims',
                  '--char_dims',
                  type=int,
                  required=True,
                  help='char embedding dimensions.')
        group.add('-char_batch_size',
                  '--char_batch_size',
                  type=int,
                  required=True,
                  help='char embedding batch size.')
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

        # entity model configurations
        group = self.parser.add_argument_group('Entity-Model')
        group.add('-entity_chunk_model',
                  '--entity_chunk_model',
                  type=str,
                  choices=["embedding_entity_chunk_model", "bert_entity_chunk_model"],
                  default="embedding_entity_chunk_model",
                  help='entity chunking model.')
        group.add('-tokens_pred_task',
                  '--tokens_pred_task',
                  type=str,
                  choices=["tokens", "wordpiece_tokens"],
                  default="tokens",
                  help='tokens prediction task.')
        group.add('-lstm_layers',
                  '--lstm_layers',
                  type=int,
                  default=1,
                  help='number of lstm layers.')
        group.add('-lstm_hidden_unit_dims',
                  '--lstm_hidden_unit_dims',
                  type=int,
                  default=128,
                  help='lstm hidden unit dimensions.')

        # realtion model configurations
        group = self.parser.add_argument_group('Relation-Model')
        group.add('-schedule_k',
                  '--schedule_k',
                  type=float,
                  default='1.0',
                  help='scheduled sampling rate.')
        group.add('-rel_kernel_sizes',
                  '--rel_kernel_sizes',
                  default=[2, 3],
                  nargs='*',
                  type=int,
                  help='relation model convolution kernel size list.')
        group.add('-rel_output_channels',
                  '--rel_output_channels',
                  type=int,
                  default=25,
                  help='relation model convolution output dimensions.')
        group.add('-ent_output_size',
                  '--ent_output_size',
                  type=int,
                  default=0,
                  help='entity model output size.')
        group.add('-context_hidden_size',
                  '--context_hidden_size',
                  type=int,
                  default=0,
                  help='relation model hidden size.')
        group.add('-rel_output_size',
                  '--rel_output_size',
                  type=int,
                  default=0,
                  help='relation model output size.')
        group.add('-span_batch_size',
                  '--span_batch_size',
                  type=int,
                  default=-1,
                  help='span convolution batch size.')

        # regularization configurations
        group = self.parser.add_argument_group('Regularization')
        group.add('-dropout',
                  '--dropout',
                  type=float,
                  default=0.5,
                  help='dropout rate.')
        
        # pretrained model
        group = self.parser.add_argument_group('Pretrained')
        group.add('-bert_model_name',
                  '--bert_model_name',
                  type=str,
                  required=False,
                  help='bert model name.')
        group.add('-bert_output_size',
                  '--bert_output_size',
                  type=int,
                  default=768,
                  help='bert output size.')
        group.add('-bert_dropout',
                  '--bert_dropout',
                  type=float,
                  default=0.1,
                  help='bert dropout rate.')
        group.add('--fine_tune',
                  '--fine_tune',
                  action='store_true',
                  help='fine-tune pretrained model.')

    def add_optimizer_cfgs(self):
        """This function adds optimizer arguements
        """

        # gradient strategy
        self.parser.add('-gradient_clipping',
                        '--gradient_clipping',
                        type=float,
                        default=1.0,
                        help='gradient clipping threshold.')

        # learning rate
        self.parser.add('--learning_rate',
                        '-learning_rate',
                        type=float,
                        default=1e-3,
                        help="Starting learning rate. "
                        "Recommended settings: sgd = 1, adagrad = 0.1, "
                        "adadelta = 1, adam = 0.001")

        # Adam configurations
        group = self.parser.add_argument_group('Adam')
        group.add('-adam_beta1',
                  '--adam_beta1',
                  type=float,
                  default=0.9,
                  help="The beta1 parameter used by Adam. "
                  "Almost without exception a value of 0.9 is used in "
                  "the literature, seemingly giving good results, "
                  "so we would discourage changing this value from "
                  "the default without due consideration.")
        group.add('-adam_beta2',
                  '--adam_beta2',
                  type=float,
                  default=0.999,
                  help='The beta2 parameter used by Adam. '
                  'Typically a value of 0.999 is recommended, as this is '
                  'the value suggested by the original paper describing '
                  'Adam, and is also the value adopted in other frameworks '
                  'such as Tensorflow and Kerras, i.e. see: '
                  'https://www.tensorflow.org/api_docs/python/tf/train/Adam'
                  'Optimizer or '
                  'https://keras.io/optimizers/ . '
                  'Whereas recently the paper "Attention is All You Need" '
                  'suggested a value of 0.98 for beta2, this parameter may '
                  'not work well for normal models / default '
                  'baselines.')
        group.add('-adam_epsilon',
                  '--adam_epsilon',
                  type=float,
                  default=1e-8,
                  help='adam epsilon')
        group.add('-adam_weight_decay_rate',
                  '--adam_weight_decay_rate',
                  type=float,
                  default=0.0,
                  help='adam weight decay rate')

    def add_run_cfgs(self):
        """This function adds running arguments
        """

        # training configurations
        group = self.parser.add_argument_group('Training')
        group.add('-seed',
                  '--seed',
                  type=int,
                  default=5216,
                  help='radom seed.')
        group.add('-epoches',
                  '--epoches',
                  type=int,
                  default=1000,
                  help='training epoches.')
        group.add('-warmup_rate',
                  '--warmup_rate',
                  type=float,
                  default=0.0,
                  help='warmup rate.')
        group.add('-early_stop',
                  '--early_stop',
                  type=int,
                  default=30,
                  help='early stop threshold.')
        group.add('-train_batch_size',
                  '--train_batch_size',
                  type=int,
                  default=100,
                  help='batch size during training.')
        group.add('-gradient_accumulation_steps',
                  '--gradient_accumulation_steps',
                  type=int,
                  default=1,
                  help='Number of updates steps to accumulate before performing a backward/update pass.')
        group.add('-continue_training',
                  '--continue_training',
                  action='store_true',
                  help='continue training from last.')

        # testing configurations
        group = self.parser.add_argument_group('Testing')
        group.add('-test_batch_size',
                  '--test_batch_size',
                  type=int,
                  default=100,
                  help='batch size during testing.')
        group.add('-validate_every',
                  '--validate_every',
                  type=int,
                  default=4000,
                  help='output result every n samples during validating.')

        # gpu configurations
        group = self.parser.add_argument_group('GPU')
        group.add('-device',
                  '--device',
                  type=int,
                  default=-1,
                  help='cpu: device = -1, gpu: gpu device id(device >= 0).')

        # logging configurations
        group = self.parser.add_argument_group('logging')
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
        group.add('-log_file',
                  '--log_file',
                  type=str,
                  action=CheckPathAction,
                  required=True,
                  help='logging file during running.')
        group.add('-file_log_level',
                  '--file_log_level',
                  type=str,
                  action=StoreLoggingLevelAction,
                  choices=StoreLoggingLevelAction.CHOICES,
                  default="NOTSET",
                  help='file logging output level.')
        group.add('-logging_steps',
                  '--logging_steps',
                  type=int,
                  default=10,
                  help='Logging every N update steps.')

    def parse_args(self):
        """This function parses arguments and initializes logger
        
        Returns:
            dict -- config arguments
        """

        cfg = self.parser.parse_args()
        init_logger(root_log_level=getattr(cfg, 'root_log_level',
                                           logging.DEBUG),
                    console_log_level=getattr(cfg, 'console_log_level',
                                              logging.NOTSET),
                    log_file=getattr(cfg, 'log_file', None),
                    log_file_level=getattr(cfg, 'log_file_level',
                                           logging.NOTSET))
        
        if not os.path.exists(cfg.save_dir):
            os.makedirs(cfg.save_dir)

        if not os.path.exists(cfg.train_model_dir):
            os.makedirs(cfg.train_model_dir)

        return cfg

    def format_values(self):
        return self.parser.format_values()
