from collections import defaultdict
import os
import random
import logging

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup

from utils.argparse import ConfigurationParer
from utils.prediction_outputs import print_predictions
from utils.eval import eval_file
from inputs.vocabulary import Vocabulary
from inputs.fields.token_field import TokenField
from inputs.fields.raw_token_field import RawTokenField
from inputs.fields.char_token_field import CharTokenField
from inputs.fields.map_token_field import MapTokenField
from inputs.instance import Instance
from inputs.datasets.dataset import Dataset
from inputs.text_readers.text_base_reader import TextBaseReader
from models.ent_models.joint_ent_model import JointEntModel
from models.ent_models.pipeline_ent_model import PipelineEntModel
from models.rel_models.context_rel_model import ConRelModel
from models.rel_models.ent_context_rel_model import EntConRelModel
from models.joint_models.joint_relation_extraction_model import JointREModel
from utils.nn_utils import get_n_trainable_parameters

logger = logging.getLogger(__name__)


def step(args, model, batch_inputs, device):
    batch_inputs["tokens"] = torch.LongTensor(batch_inputs["tokens"])
    batch_inputs["char_tokens"] = torch.LongTensor(batch_inputs["char_tokens"])
    if args.entity_model == 'joint':
        batch_inputs["entity_labels"] = torch.LongTensor(batch_inputs["entity_labels"])
    else:
        batch_inputs["entity_span_labels"] = torch.LongTensor(batch_inputs["entity_span_labels"])
    batch_inputs["tokens_mask"] = torch.LongTensor(batch_inputs["tokens_mask"])

    if args.embedding_model == 'pretrained':
        batch_inputs["subword_tokens"] = torch.LongTensor(batch_inputs["subword_tokens"])
        batch_inputs["subword_tokens_index"] = torch.LongTensor(batch_inputs["subword_tokens_index"])
        batch_inputs["subword_segment_ids"] = torch.LongTensor(batch_inputs["subword_segment_ids"])

    if device > -1:
        batch_inputs["tokens"] = batch_inputs["tokens"].cuda(device=device, non_blocking=True)
        batch_inputs["char_tokens"] = batch_inputs["char_tokens"].cuda(device=device, non_blocking=True)
        if args.entity_model == 'joint':
            batch_inputs["entity_labels"] = batch_inputs["entity_labels"].cuda(device=device, non_blocking=True)
        else:
            batch_inputs["entity_span_labels"] = batch_inputs["entity_span_labels"].cuda(device=device,
                                                                                         non_blocking=True)
        batch_inputs["tokens_mask"] = batch_inputs["tokens_mask"].cuda(device=device, non_blocking=True)

        if args.embedding_model == 'pretrained':
            batch_inputs["subword_tokens"] = batch_inputs["subword_tokens"].cuda(device=device, non_blocking=True)
            batch_inputs["subword_tokens_index"] = batch_inputs["subword_tokens_index"].cuda(device=device,
                                                                                             non_blocking=True)
            batch_inputs["subword_segment_ids"] = batch_inputs["subword_segment_ids"].cuda(device=device,
                                                                                           non_blocking=True)

    outputs = model(batch_inputs)
    batch_outputs = []
    for sent_idx in range(len(batch_inputs['tokens_lens'])):
        sent_output = dict()
        sent_output['tokens'] = batch_inputs['tokens'][sent_idx].cpu().numpy()
        if args.entity_model == 'joint':
            sent_output["sequence_labels"] = batch_inputs["entity_labels"][sent_idx].cpu().numpy()
        else:
            sent_output["sequence_labels"] = batch_inputs["entity_span_labels"][sent_idx].cpu().numpy()
        sent_output['span2ent'] = batch_inputs['span2ent'][sent_idx]
        sent_output['span2rel'] = batch_inputs['span2rel'][sent_idx]
        sent_output['seq_len'] = batch_inputs['tokens_lens'][sent_idx]
        sent_output["sequence_label_preds"] = outputs['sequence_label_preds'][sent_idx].cpu().numpy()
        sent_output['all_ent_preds'] = outputs['all_ent_preds'][sent_idx]
        sent_output['all_rel_preds'] = outputs['all_rel_preds'][sent_idx]
        batch_outputs.append(sent_output)

    return batch_outputs, outputs['ent_loss'], outputs['rel_loss']


def train(args, dataset, model):
    logger.info("Training starting...")

    for name, param in model.named_parameters():
        logger.info("{!r}: size: {} requires_grad: {}.".format(name, param.size(), param.requires_grad))

    logger.info("Trainable parameters size: {}.".format(get_n_trainable_parameters(model)))

    parameters = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = []
    for name, param in parameters:
        params = {'params': [param]}
        if any(item in name for item in no_decay):
            params['weight_decay_rate'] = 0.0

        if 'pretrained' in name:
            params['lr'] = args.ptm_learning_rate

        optimizer_grouped_parameters.append(params)

    optimizer = AdamW(optimizer_grouped_parameters,
                      betas=(args.adam_beta1, args.adam_beta2),
                      lr=args.learning_rate,
                      eps=args.adam_epsilon,
                      weight_decay=args.adam_weight_decay_rate,
                      correct_bias=False)

    total_train_steps = (dataset.get_dataset_size("train") + args.train_batch_size * args.gradient_accumulation_steps -
                         1) / (args.train_batch_size * args.gradient_accumulation_steps) * args.epochs
    num_warmup_steps = int(args.warmup_rate * total_train_steps) + 1
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=total_train_steps)

    last_epoch = 1
    batch_id = 0
    best_f1 = 0.0
    early_stop_cnt = 0
    accumulation_steps = 0
    model.zero_grad()

    if args.embedding_model == 'word_char' or args.lstm_layers > 0:
        sort_key = "tokens"
    else:
        sort_key = None

    for epoch, batch in dataset.get_batch('train', args.train_batch_size, sort_key):

        if last_epoch != epoch or (batch_id != 0 and batch_id % args.validate_every == 0):
            if accumulation_steps != 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            if epoch > args.pretrain_epochs:
                dev_f1 = dev(args, dataset, model)

                if dev_f1 > best_f1:
                    early_stop_cnt = 0
                    best_f1 = dev_f1
                    logger.info("Save model...")
                    test(args, dataset, model)

                    torch.save(
                        model.state_dict(),
                        open(
                            os.path.join(args.model_checkpoints_dir,
                                         "epoch_{}_batch_{}_{:04.2f}".format(last_epoch, batch_id, 100 * best_f1)),
                            "wb",
                        ),
                    )
                    torch.save(model.state_dict(), open(args.best_model_path, "wb"))
                elif last_epoch != epoch:
                    early_stop_cnt += 1
                    if early_stop_cnt > args.early_stop:
                        logger.info("Early Stop: best F1 score: {:6.2f}%".format(100 * best_f1))
                        break
        if epoch > args.epochs:
            torch.save(model.state_dict(), open(args.last_model_path, "wb"))
            logger.info("Training Stop: best F1 score: {:6.2f}%".format(100 * best_f1))
            break

        if last_epoch != epoch:
            batch_id = 0
            last_epoch = epoch

        model.train()
        batch_id += len(batch['tokens_lens'])
        batch['epoch'] = (epoch - 1)
        _, ent_loss, rel_loss = step(args, model, batch, args.device)
        loss = ent_loss + rel_loss
        if batch_id % args.logging_steps == 0:
            logger.info("Epoch: {} Batch: {} Loss: {} (Ent_loss: {} Rel_loss: {})".format(
                epoch, batch_id, loss.item(), ent_loss.item(), rel_loss.item()))

        if args.gradient_accumulation_steps > 1:
            loss /= args.gradient_accumulation_steps

        loss.backward()

        accumulation_steps = (accumulation_steps + 1) % args.gradient_accumulation_steps
        if accumulation_steps == 0:
            nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=args.gradient_clipping)
            optimizer.step()
            scheduler.step()
            model.zero_grad()

    state_dict = torch.load(open(args.best_model_path, "rb"), map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    test(args, dataset, model)


def dev(args, dataset, model):
    logger.info("Validate starting...")
    model.zero_grad()

    all_outputs = []
    all_ent_loss = []
    all_rel_loss = []

    if args.embedding_model == 'word_char' or args.lstm_layers > 0:
        sort_key = "tokens"
    else:
        sort_key = None

    for _, batch in dataset.get_batch('dev', args.test_batch_size, sort_key):
        model.eval()
        with torch.no_grad():
            batch_outpus, ent_loss, rel_loss = step(args, model, batch, args.device)
        all_outputs.extend(batch_outpus)
        all_ent_loss.append(ent_loss.item())
        all_rel_loss.append(rel_loss.item())
    mean_ent_loss = np.mean(all_ent_loss)
    mean_rel_loss = np.mean(all_rel_loss)
    mean_loss = mean_ent_loss + mean_rel_loss

    logger.info("Validate Avgloss: {} (Ent_loss: {} Rel_loss: {})".format(mean_loss, mean_ent_loss, mean_rel_loss))

    dev_output_file = os.path.join(args.save_dir, "dev.output")
    print_predictions(all_outputs, dev_output_file, dataset.vocab,
                      'entity_labels' if args.entity_model == 'joint' else 'entity_span_labels')
    eval_metrics = ['token', 'span', 'ent', 'rel', 'exact-rel']
    token_score, span_score, ent_score, rel_score, exact_rel_score = eval_file(dev_output_file, eval_metrics)
    return ent_score + exact_rel_score


def test(args, dataset, model):
    logger.info("Testing starting...")
    model.zero_grad()

    all_outputs = []

    if args.embedding_model == 'word_char' or args.lstm_layers > 0:
        sort_key = "tokens"
    else:
        sort_key = None

    for _, batch in dataset.get_batch('test', args.test_batch_size, sort_key):
        model.eval()
        with torch.no_grad():
            batch_outpus, ent_loss, rel_loss = step(args, model, batch, args.device)
        all_outputs.extend(batch_outpus)
    test_output_file = os.path.join(args.save_dir, "test.output")
    print_predictions(all_outputs, test_output_file, dataset.vocab,
                      'entity_labels' if args.entity_model == 'joint' else 'entity_span_labels')
    eval_metrics = ['token', 'span', 'ent', 'rel', 'exact-rel']
    eval_file(test_output_file, eval_metrics)


def main():
    # config settings
    parser = ConfigurationParer()
    parser.add_input_args()
    parser.add_model_args()
    parser.add_optimizer_args()
    parser.add_run_args()

    args = parser.parse_args()
    logger.info(parser.format_values())

    # set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device > -1 and not torch.cuda.is_available():
        logger.error('config conflicts: no gpu available, use cpu for training.')
        args.device = -1
    if args.device > -1:
        torch.cuda.manual_seed(args.seed)

    # define fields
    tokens = TokenField("tokens", "tokens", "tokens", True)
    raw_tokens = RawTokenField("raw_tokens", "tokens")
    char_tokens = CharTokenField("char_tokens", "char_tokens", "tokens", True)
    entity_span_labels = TokenField("entity_span_labels", "entity_span_labels", "entity_span_labels", True)
    entity_labels = TokenField("entity_labels", "entity_labels", "entity_labels", True)
    span2ent = MapTokenField("span2ent", "span2ent", "span2ent", True)
    span2rel = MapTokenField("span2rel", "span2rel", "span2rel", True)
    subword_tokens = TokenField("subword_tokens", "subword", "subword_tokens", False)
    subword_tokens_index = RawTokenField("subword_tokens_index", "subword_tokens_index")
    subword_segment_ids = RawTokenField("subword_segment_ids", "subword_segment_ids")
    fields = [tokens, raw_tokens, char_tokens, entity_span_labels, entity_labels, span2ent, span2rel]

    if args.embedding_model == 'pretrained':
        fields.extend([subword_tokens, subword_tokens_index, subword_segment_ids])

    # define counter and vocabulary
    counter = defaultdict(lambda: defaultdict(int))
    vocab = Vocabulary()

    # define instance
    train_instance = Instance(fields)
    dev_instance = Instance(fields)
    test_instance = Instance(fields)

    # define dataset reader
    max_len = {'tokens': args.max_sent_len, 'subword_tokens': args.max_subword_len}
    tokenizers = {}
    pretrained_vocab = {}
    if args.embedding_model == 'pretrained':
        ptm_tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name, do_lower_case=args.low_case)
        logger.info("Load pretrained models' tokenizer successfully.")
        tokenizers['subword'] = ptm_tokenizer.tokenize
        pretrained_vocab['subword'] = ptm_tokenizer.vocab
    ace_train_reader = TextBaseReader(args.train_file, False, args.low_case, max_len, args.entity_schema)
    ace_dev_reader = TextBaseReader(args.dev_file, False, args.low_case, max_len, args.entity_schema)
    ace_test_reader = TextBaseReader(args.test_file, False, args.low_case, max_len, args.entity_schema)

    # define dataset
    ace_dataset = Dataset("DEMO")
    ace_dataset.add_instance("train", train_instance, ace_train_reader, is_count=True, is_train=True)
    ace_dataset.add_instance("dev", dev_instance, ace_dev_reader, is_count=True, is_train=False)
    ace_dataset.add_instance("test", test_instance, ace_test_reader, is_count=True, is_train=False)

    min_count = {
        "tokens": 1,
        "char_tokens": 1,
        "entity_span_labels": 1,
        "entity_labels": 1,
        "span2ent": 1,
        "span2rel": 1,
    }
    # note: no_pad_namespace and no_unk_namespace is for vocabulary namespace
    no_pad_namespace = ["span2ent", "span2rel"]
    no_unk_namespace = ["entity_span_labels", "entity_labels", "span2ent", "span2rel"]
    tokens_to_add = {"span2ent": ["None"], "span2rel": ["None"]}
    contain_pad_namespace = {"subword": "[PAD]", "entity_span_labels": "O", "entity_labels": "O"}
    contain_unk_namespace = {"subword": "[UNK]"}
    ace_dataset.build_dataset(vocab=vocab,
                              counter=counter,
                              min_count=min_count,
                              pretrained_vocab=pretrained_vocab,
                              no_pad_namespace=no_pad_namespace,
                              no_unk_namespace=no_unk_namespace,
                              contain_pad_namespace=contain_pad_namespace,
                              contain_unk_namespace=contain_unk_namespace,
                              tokens_to_add=tokens_to_add)
    wo_padding_namespace = ["raw_tokens", "span2ent", "span2rel"]
    ace_dataset.set_wo_padding_namespace(wo_padding_namespace=wo_padding_namespace)

    if args.test:
        vocab = Vocabulary.load(args.vocabulary_file)
    else:
        vocab.save(args.vocabulary_file)

    # entity model
    if args.entity_model == 'joint':
        ent_model = JointEntModel(args, vocab)
        rel_model = EntConRelModel(args, vocab, ent_model.get_hidden_size())
    else:
        ent_model = PipelineEntModel(args, vocab)
        rel_model = ConRelModel(args, vocab, ent_model.get_hidden_size(), ent_model.get_ent_span_feature_size())

    # joint model
    model = JointREModel(args=args, ent_model=ent_model, rel_model=rel_model, vocab=vocab)

    # continue training
    if args.continue_training and os.path.exists(args.last_model_path):
        state_dict = torch.load(open(args.last_model_path, 'rb'), map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        logger.info("Loading last training model {} successfully.".format(args.last_model_path))

    if args.test and os.path.exists(args.best_model_path):
        state_dict = torch.load(open(args.best_model_path, 'rb'), map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        logger.info("Loading best training model {} successfully for testing.".format(args.best_model_path))

    if args.device > -1:
        model.cuda(device=args.device)

    if args.test:
        test(args, ace_dataset, model)
    else:
        train(args, ace_dataset, model)


if __name__ == '__main__':
    main()
