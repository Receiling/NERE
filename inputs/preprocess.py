import sys
import json
import os
from concurrent import futures
from collections import defaultdict
import json

import spacy
import fire
from pytorch_transformers import BertTokenizer

sys.path.append('../')
from utils.entity_chunking import get_entity_span


def conll03_preprocess(source_file, target_file):
    with open(source_file, 'r') as fin, open(target_file, 'w') as fout:
        sentId = 0
        tokens = []
        ent_labels = []
        for line in fin:
            line = line.strip()
            if len(line) == 0:
                span2ent = get_entity_span(ent_labels)
                ents = []
                for span, ent_label in span2ent.items():
                    ents.append({
                        'emId': len(ents),
                        'text': ' '.join(tokens[span[0]:span[1]]),
                        'offset': span,
                        'label': ent_label
                    })
                sent = json.dumps({
                    'sentId': sentId,
                    'sentText': ' '.join(tokens),
                    'articleId': sentId,
                    'entityMentions': ents,
                    'relationMentions': []
                })
                print(sent, file=fout)
                tokens = []
                ent_labels = []
                sentId += 1
                continue

            items = line.split()
            tokens.append(items[0])
            ent_labels.append(items[3])


def multi_threads_entity_tagger(source_file_dir, num_workers):
    entity2tokens = defaultdict(list)

    # Load English tokenizer, tagger, parser, NER and word vectors
    nlp = spacy.load("en_core_web_md")
    print("Load `en_core_web_md` model successfully.")

    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
    print("Load bert tokenizer successfully.")

    def entity_tagger(source_file):
        target_file = source_file + '_preprocessed'
        with open(source_file, 'r', encoding='utf-8') as fin, open(target_file,
                                                                   'w',
                                                                   encoding='utf-8') as fout:
            sentId = 0
            for line in fin:
                line = line.strip()
                if len(line) <= 0:
                    continue
                sent = nlp(line)
                tokens = list(map(str, sent))

                ent_labels = []
                for token in sent:
                    if token.ent_iob_ == 'O':
                        ent_labels.append(str(token.ent_iob_))
                    else:
                        ent_labels.append(str(token.ent_iob_) + '-' + str(token.ent_type_))

                span2ent = get_entity_span(ent_labels)

                wordpiece_tokens = ['[CLS]']
                wordpiece_tokens_index = []
                cur_pos = 1
                for token in tokens:
                    tokenized_token = list(bert_tokenizer.tokenize(token))
                    wordpiece_tokens.extend(tokenized_token)
                    wordpiece_tokens_index.append([cur_pos, cur_pos + len(tokenized_token)])
                    cur_pos += len(tokenized_token)
                wordpiece_tokens.append('[SEP]')

                ents = []
                for span, ent_label in span2ent.items():
                    ents.append({
                        'emId': len(ents),
                        'text': ' '.join(tokens[span[0]:span[1]]),
                        'offset': span,
                        'label': ent_label
                    })
                    entity2tokens[ent_label + '#' +
                                  str(wordpiece_tokens_index[span[1] - 1][1] -
                                      wordpiece_tokens_index[span[0]][0])].append(' '.join(
                                          wordpiece_tokens[wordpiece_tokens_index[span[0]][0]:
                                                           wordpiece_tokens_index[span[1] - 1][1]]))

                sent = json.dumps({
                    'sentId': sentId,
                    'sentText': ' '.join(tokens),
                    'wordpieceSentText': ' '.join(wordpiece_tokens),
                    'wordpieceTokensIndex': wordpiece_tokens_index,
                    'articleId': sentId,
                    'entityMentions': ents,
                    'relationMentions': []
                })
                print(sent, file=fout)

                sentId += 1
                if sentId % 10000 == 0:
                    print("processing {} sentences of file {}.".format(sentId, source_file))

        print("File {} is preprocessed completely, processing all {} sentences.".format(
            source_file, sentId))

    source_files = [
        os.path.join(source_file_dir, source_file) for source_file in os.listdir(source_file_dir)
        if not os.path.isdir(source_file)
    ]
    num_workers = min(num_workers, len(source_files))
    print("The number of workers: {}.".format(num_workers))
    with futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        executor.map(entity_tagger, source_files)

    print("All files in dictionary {} are preprocessed completely.".format(source_file_dir))

    entity2tokens_path = os.path.join(source_file_dir, 'entity2tokens.json')
    json.dump(entity2tokens, open(entity2tokens_path, 'w'))
    print('Save entity2tokens as {}.'.format(entity2tokens_path))


if __name__ == '__main__':
    fire.Fire({'conll2003': conll03_preprocess, 'entity_tagger': multi_threads_entity_tagger})
