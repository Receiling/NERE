def print_predictions(outputs, file_path, vocab, sequence_label_domain=None):
    """Prints prediction results
    
    Args:
        outputs (list): prediction outputs
        file_path (str): output file path
        vocab (Vocabulary): vocabulary
        sequence_label_domain (str, optional): sequence label domain. Defaults to None.
    """

    with open(file_path, 'w') as fout:
        for sent_output in outputs:
            seq_len = sent_output['seq_len']
            assert 'tokens' in sent_output
            tokens = [vocab.get_token_from_index(token, 'tokens') for token in sent_output['tokens'][:seq_len]]
            print("Token\t{}".format(' '.join(tokens)), file=fout)

            if 'text' in sent_output:
                print(f"Text\t{sent_output['text']}", file=fout)

            if 'sequence_labels' in sent_output and 'sequence_label_preds' in sent_output:
                sequence_labels = [
                    vocab.get_token_from_index(true_sequence_label, sequence_label_domain)
                    for true_sequence_label in sent_output['sequence_labels'][:seq_len]
                ]
                sequence_label_preds = [
                    vocab.get_token_from_index(pred_sequence_label, sequence_label_domain)
                    for pred_sequence_label in sent_output['sequence_label_preds'][:seq_len]
                ]

                print("Sequence-Label-True\t{}".format(' '.join(sequence_labels)), file=fout)
                print("Sequence-Label-Pred\t{}".format(' '.join(sequence_label_preds)), file=fout)

            if 'span2ent' in sent_output:
                for span, ent in sent_output['span2ent'].items():
                    ent = vocab.get_token_from_index(ent, 'span2ent')
                    assert ent != 'None', "true relation can not be `None`."

                    print("Ent-True\t{}\t{}\t{}".format(ent, span, ' '.join(tokens[span[0]:span[1]])), file=fout)

            if 'all_ent_preds' in sent_output:
                for span, ent in sent_output['all_ent_preds'].items():
                    # ent = vocab.get_token_from_index(ent, 'span2ent')

                    print("Ent-Span-Pred\t{}".format(span), file=fout)
                    print("Ent-Pred\t{}\t{}\t{}".format(ent, span, ' '.join(tokens[span[0]:span[1]])), file=fout)

            if 'span2rel' in sent_output:
                for (span1, span2), rel in sent_output['span2rel'].items():
                    rel = vocab.get_token_from_index(rel, 'span2rel')
                    assert rel != 'None', "true relation can not be `None`."

                    if rel[-1] == '<':
                        span1, span2 = span2, span1
                    print("Rel-True\t{}\t{}\t{}\t{}\t{}".format(rel[:-2], span1, span2,
                                                                ' '.join(tokens[span1[0]:span1[1]]),
                                                                ' '.join(tokens[span2[0]:span2[1]])),
                          file=fout)

            if 'all_rel_preds' in sent_output:
                for (span1, span2), rel in sent_output['all_rel_preds'].items():
                    # rel = vocab.get_token_from_index(rel, 'span2rel')

                    if rel[-1] == '<':
                        span1, span2 = span2, span1
                    print("Rel-Pred\t{}\t{}\t{}\t{}\t{}".format(rel[:-2], span1, span2,
                                                                ' '.join(tokens[span1[0]:span1[1]]),
                                                                ' '.join(tokens[span2[0]:span2[1]])),
                          file=fout)

            print(file=fout)
