from abc import ABC, abstractclassmethod


class Field(ABC):
    """Abstract class `Field` define one indexing method,
    generate counter from raw text data and index token in raw text data

    Arguments:
        ABC {ABC} -- abstract base class
    """
    @abstractclassmethod
    def count_vocab_items(self, counter, sentences):
        """Constructs counter using each sentence content,
        prepare for vocabulary

        Arguments:
            counter {dict} -- element count dict
            sentences {list} -- text data
        """

        raise NotImplementedError

    @abstractclassmethod
    def index(self, instance, vocab, sentences):
        """Constrcuts instance using sentences and vocabulary,
        each namespace is a mapping method using different type data

        Arguments:
            instance {dict} -- collections of various fields
            vocab {dict} -- vocabulary
            sentences {list} -- text data
        """

        raise NotImplementedError
