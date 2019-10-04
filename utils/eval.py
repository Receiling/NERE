from collections import defaultdict, namedtuple

from utils.entity_chunking import get_entity_span
from utils.logging import logger


class EvalCounts():
    """This class is evaluating counters
    """
    
    def __init__(self):
        self.pred_correct_cnt = 0
        self.correct_cnt = 0
        self.pred_cnt = 0

        self.pred_correct_types_cnt = defaultdict(int)
        self.correct_types_cnt = defaultdict(int)
        self.pred_types_cnt = defaultdict(int)


def eval_file(file_path):
    """This function evaluates results file
    
    Arguments:
        file_path {str} -- file path
    
    Returns:
        tuple -- ent_score, rel_score
    """
    
    with open(file_path, 'r') as fin:
        sents = []
        sent = [[], [], [], [], [], []]
        for line in fin:
            line = line.strip('\r\n')
            if line == "":
                if len(sent[0]) != 0:
                    sents.append(sent)
                    sent = [[], [], [], [], [], []]
            else:
                words = line.split('\t')
                if len(words) == 3:
                    if words[0] == 'Ent-True':
                        sent[2].append([words[1], eval(words[2])])
                    elif words[0] == 'Ent-Pred':
                        sent[3].append([words[1], eval(words[2])])
                    else:
                        sent[0].append(words[1])
                        sent[1].append(words[2])
                elif len(words) == 4:
                    if words[0] == 'Rel-True':
                        sent[4].append([words[1], eval(words[2]), eval(words[3])])
                    elif words[0] == 'Rel-Pred':
                        sent[5].append([words[1], eval(words[2]), eval(words[3])])
        if len(sent[0]) != 0:
            sents.append(sent)

    token_counts = EvalCounts()
    ent_counts = EvalCounts()
    rel_counts = EvalCounts()

    for sent in sents:
        evaluate(sent, token_counts, ent_counts, rel_counts)

    logger.info("------------------------------Token------------------------------")
    token_score = report(token_counts)
    logger.info("------------------------------Entity------------------------------")
    ent_score = report(ent_counts)
    logger.info("-----------------------------Relation-----------------------------")
    rel_score = report(rel_counts)
    logger.info("------------------------------------------------------------------")

    return ent_score, rel_score


def evaluate(sent, token_counts, ent_counts, rel_counts):
    """This function calculate counters
    
    Arguments:
        sent {list} -- line
        token_counts {dict} -- token counters
        ent_counts {dict} -- entity counters
        rel_counts {dict} -- relation counters
    """
    
    # evaluate token
    for token1, token2 in zip(sent[0], sent[1]):
        if token1 != 'O':
            token_counts.correct_cnt += 1
            token_counts.correct_types_cnt[token1] += 1
        if token2 != 'O':
            token_counts.pred_cnt += 1
            token_counts.pred_types_cnt[token2] += 1
        if token1 == token2 and token1 != 'O':
            token_counts.pred_correct_cnt += 1
            token_counts.pred_correct_types_cnt[token1] += 1
    
    # evaluate entity
    correct_ent2idx = defaultdict(set)
    correct_span2ent = dict()
    for ent, span in sent[2]:
        correct_span2ent[span] = ent
        correct_ent2idx[ent].add(span)
    
    pred_ent2idx = defaultdict(set)
    pred_span2ent = dict()
    for ent, span in sent[3]:
        pred_span2ent[span] = ent
        pred_ent2idx[ent].add(span)
    
    all_ents = set(correct_ent2idx) | set(pred_ent2idx)
    for ent in all_ents:
        ent_counts.correct_cnt += len(correct_ent2idx[ent])
        ent_counts.correct_types_cnt[ent] += len(correct_ent2idx[ent])
        ent_counts.pred_cnt += len(pred_ent2idx[ent])
        ent_counts.pred_types_cnt[ent] += len(pred_ent2idx[ent])
        pred_correct_cnt = len(correct_ent2idx[ent] & pred_ent2idx[ent])
        ent_counts.pred_correct_cnt += pred_correct_cnt
        ent_counts.pred_correct_types_cnt[ent] += pred_correct_cnt

    # evaluate relation
    correct_rel2idx = defaultdict(set)
    for rel, span1, span2 in sent[4]:
        if span1 not in correct_span2ent or span2 not in correct_span2ent:
            continue
        correct_rel2idx[rel].add((span1, correct_span2ent[span1], span2, correct_span2ent[span2]))

    pred_rel2idx = defaultdict(set)
    for rel, span1, span2 in sent[5]:
        pred_rel2idx[rel].add((span1, pred_span2ent[span1], span2, pred_span2ent[span2]))

    all_rels = set(correct_rel2idx) | set(pred_rel2idx)
    for rel in all_rels:
        rel_counts.correct_cnt += len(correct_rel2idx[rel])
        rel_counts.correct_types_cnt[rel] += len(correct_rel2idx[rel])
        rel_counts.pred_cnt += len(pred_rel2idx[rel])
        rel_counts.pred_types_cnt[rel] += len(pred_rel2idx[rel])
        pred_correct_rel_cnt = len(correct_rel2idx[rel] & pred_rel2idx[rel])
        rel_counts.pred_correct_cnt += pred_correct_rel_cnt
        rel_counts.pred_correct_types_cnt[rel] += pred_correct_rel_cnt


def report(counts):
    """This function print evaluation results
    
    Arguments:
        counts {dict} -- counters
    
    Returns:
        float -- f1 score
    """
    
    p, r, f = calculate_metrics(counts.pred_correct_cnt, counts.pred_cnt, counts.correct_cnt)
    logger.info("truth cnt: {} pred cnt: {} correct cnt: {}".format(counts.correct_cnt,
                                                                    counts.pred_cnt,
                                                                    counts.pred_correct_cnt))
    logger.info("precision: {:6.2f}%".format(100 * p))
    logger.info("recall: {:6.2f}%".format(100 * r))
    logger.info("f1: {:6.2f}%".format(100 * f))

    score = f

    for type in counts.pred_correct_types_cnt:
        p, r, f = calculate_metrics(counts.pred_correct_types_cnt[type],
                                    counts.pred_types_cnt[type],
                                    counts.correct_types_cnt[type])
        logger.info("--------------------------------------------------")
        logger.info("type: {:17}".format(type))
        logger.info("truth cnt: {} pred cnt: {} correct cnt: {}".format(
            counts.correct_types_cnt[type],
            counts.pred_types_cnt[type],
            counts.pred_correct_types_cnt[type]))
        logger.info("precision: {:6.2f}%".format(100 * p))
        logger.info("recall: {:6.2f}%".format(100 * r))
        logger.info("f1: {:6.2f}%".format(100 * f))

    return score


def calculate_metrics(pred_correct_cnt, pred_cnt, correct_cnt):
    """This function calculation metrics: precision, recall, f1-score
    
    Arguments:
        pred_correct_cnt {int} -- the number of corrected prediction
        pred_cnt {int} -- the number of prediction
        correct_cnt {int} -- the numbert of truth
    
    Returns:
        tuple -- precision, recall, f1-score
    """
    
    tp, fp, fn = pred_correct_cnt, pred_cnt - pred_correct_cnt, correct_cnt - pred_correct_cnt
    p = 0 if tp + fp == 0 else(tp / (tp + fp))
    r = 0 if tp + fn == 0 else(tp / (tp + fn))
    f = 0 if p + r == 0 else(2 * p * r / (p + r))
    return p, r, f
