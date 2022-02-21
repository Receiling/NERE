## Dataset Processing

### ACE2005([https://catalog.ldc.upenn.edu/LDC2006T06](https://catalog.ldc.upenn.edu/LDC2006T06))

Firstly, achieving **ACE2005** dataset with scripts provided by [two-are-better-than-one](https://github.com/LorrinWWW/two-are-better-than-one/tree/master/datasets).
Then, getting the final data by running [`ace2005.sh`](https://github.com/Receiling/NERE/tree/master/data/ace2005.sh)
```bash
./ace2005.sh ace2005_folder
```
Note that we adopt `bert-base-uncased` tokenizer as the default. 
You can try other tokenizers by modifying `ace2005.sh`.

### SciERC([http://nlp.cs.washington.edu/sciIE/](http://nlp.cs.washington.edu/sciIE/))

Firstly, downloading **SciERC** dataset from [sciIE](http://nlp.cs.washington.edu/sciIE/).
Then, getting the final data by running [`scierc.sh`](https://github.com/Receiling/NERE/tree/master/data/scierc.sh)
```bash
./scierc.sh scierc_folder
```
Note that we adopt `bert-base-uncased` tokenizer as the default.
You can try other tokenizers by modifying `scierc.sh`.

### Demo
We provide demo samples in the folder [`demo/`](https://github.com/Receiling/NERE/tree/master/data/demo/).