import json
import fire

from transformers import AutoTokenizer


def process(source_file, target_file, pretrained_model):
    """Tokenizes sentences with pretrained tokenizers like BertTokenizer.
    This function returns BERT like input format: [CLS] xxx [SEP].
    """

    auto_tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    print("Load {} tokenizer successfully.".format(pretrained_model))

    with open(source_file, 'r', encoding='utf-8') as fin, open(target_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            sent = json.loads(line)
            tokens = sent['sentText'].split(' ')
            subword_tokens = ['[CLS]']
            subword_tokens_index = []
            cur_pos = 1
            for token in tokens:
                tokenized_token = list(auto_tokenizer.tokenize(token))
                subword_tokens.extend(tokenized_token)
                subword_tokens_index.append([cur_pos, cur_pos + len(tokenized_token)])
                cur_pos += len(tokenized_token)
            subword_tokens.append('[SEP]')
            sent['subwordSentText'] = ' '.join(subword_tokens)
            sent['subwordTokensIndex'] = subword_tokens_index
            sent['subwordSegmentIds'] = [0] * len(subword_tokens)
            print(json.dumps(sent), file=fout)


if __name__ == '__main__':
    fire.Fire({"process": process})
