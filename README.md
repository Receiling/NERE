# NERE
A Universal Toolkit for Neural Entity Relation Extraction.

## Requirements
* `python`: 3.7.6
* `pytorch`: 1.8.1
* `transformers`: 4.2.2
* `configargparse`: 1.2.3
* `bidict`: 0.20.0
* `fire`: 0.3.1

## Datasets
We provide scripts and instructions for processing two datasets (ACE2005,SciERC) and demo samples in the [`data/`](https://github.com/Receiling/NERE/tree/master/data).

## I/O
We decouple complex entity relation annotation into `Field` form.
Each `Field` automatically constructs vocabulary and gets numerical input from the text.

## Models
We jointly extract entities and relation by sharing parameters.
The entity model is based on the sequence labeling framework, and the relation model is based on CNN network.
This toolkit supports both the LSTM and pre-trained model as the sentence encoder.
Besides, we implement two paradigm:
+ Entity based: first detects entities then recognizes relations between entities.
+ Span based: first detects entity spans then recognizes entities and relations.

## Training
### LSTM + Entity based
```bash
python entity_relation_extractor.py \
    --config_file config.yml \
    --save_dir ckpt/lstm_entity \
    --embedding_model word_char \
    --entity_model joint \
    --lstm_layers 1 \
    --lstm_hidden_unit_dims 256 \
    --epochs 300 \
    --device 0
```

### LSTM + Span based
```bash
python entity_relation_extractor.py \
    --config_file config.yml \
    --save_dir ckpt/lstm_span \
    --embedding_model word_char \
    --entity_model pipeline \
    --lstm_layers 1 \
    --lstm_hidden_unit_dims 256 \
    --epochs 300 \
    --device 0
```

### Pre-trained model + Entity based
```bash
python entity_relation_extractor.py \
    --config_file config.yml \
    --save_dir ckpt/ptm_entity \
    --embedding_model pretrained \
    --entity_model joint \
    --pretrained_model_name bert-base-uncased \
    --epochs 200 \
    --fine_tune \
    --device 0
```

### Pre-trained model + Span based
```bash
python entity_relation_extractor.py \
    --config_file config.yml \
    --save_dir ckpt/ptm_span \
    --embedding_model pretrained \
    --entity_model pipeline \
    --pretrained_model_name bert-base-uncased \
    --epochs 200 \
    --fine_tune \
    --device 0
```

## Inference
Note that `save_dir` should contain the trained `best_model`.

### LSTM + Entity based
```bash
python entity_relation_extractor.py \
    --config_file config.yml \
    --save_dir ckpt/lstm_entity \
    --embedding_model word_char \
    --entity_model joint \
    --lstm_layers 1 \
    --lstm_hidden_unit_dims 256 \
    --device 0 \
    --log_file test.log \
    --test
```

### LSTM + Span based
```bash
python entity_relation_extractor.py \
    --config_file config.yml \
    --save_dir ckpt/lstm_span \
    --embedding_model word_char \
    --entity_model pipeline \
    --lstm_layers 1 \
    --lstm_hidden_unit_dims 256 \
    --device 0 \
    --log_file test.log \
    --test
```

### Pre-trained model + Entity based
```bash
python entity_relation_extractor.py \
    --config_file config.yml \
    --save_dir ckpt/ptm_entity \
    --embedding_model pretrained \
    --entity_model joint \
    --pretrained_model_name bert-base-uncased \
    --fine_tune \
    --device 0 \
    --log_file test.log \
    --test
```

### Pre-trained model + Span based
```bash
python entity_relation_extractor.py \
    --config_file config.yml \
    --save_dir ckpt/ptm_span \
    --embedding_model pretrained \
    --entity_model pipeline \
    --pretrained_model_name bert-base-uncased \
    --fine_tune \
    --device 0 \
    --log_file test.log \
    --test
```

## Evaluation
This toolkit supports different evaluation criteria:
+ Entity: P/R/F1
+ Span: P/R/F1
+ Relation (relax): P/R/F1
+ Relation (strict): P/R/F1

## Notifications
+ If **OOM** occurs, we suggest that reducing `train_batch_size` and increasing `gradient_accumulation_steps` (`gradient_accumulation_steps` is used to perform *Gradient Accumulation*). 
+ This toolkit currently does not support training on multi-GPUs.
