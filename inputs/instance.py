from utils.logging import logger


class Instance():
    """ `Instance` is the collection of multiple `Field`
    """

    def __init__(self, fields, pretrained_namespace=list()):
        """This function initializes instance

        Arguments:
            fields {list} -- field list

        Keyword Arguments:
            pretrained_namespace {list} -- pretrained namespace list (default: {list()})
        """

        self.fields = list(fields)
        self.pretrained_namespace = set(pretrained_namespace)
        self.instance = {}
        for field in self.fields:
            self.instance[field.namespace] = []

    def __getitem__(self, namespace):
        if namespace not in self.instance:
            logger.error(
                "can not find the namespace {} in instance.".format(namespace))
            raise RuntimeError(
                "can not find the namespace {} in instance.".format(namespace))
        else:
            self.instance.get(namespace, None)

    def __iter__(self):
        return iter(self.fields)

    def __len__(self):
        return len(self.fields)

    def add_fields(self, fields, pretrained_namespace):
        """This function adds fields to instance

        Arguments:
            field {Field} -- field list
            pretrained_namespace {list} -- pretrained namespace list
        """

        for field in fields:
            if field.namesapce not in self.instance:
                self.fields.append(field)
                self.instance[field.namesapce] = []
            else:
                logger.warning('Field {} has been added before.'.format(
                    field.name))
        self.pretrained_namespace.update(set(pretrained_namespace))

    def add_namespace(self, namespace):
        """This function adds namespace to instance only,
        but not add to filed list

        Arguments:
            namespace {str} -- namespace name
        """

        self.instance[namespace] = []

    def count_vocab_items(self, counter, sentences):
        """This funtion constructs multiple namespace in counter

        Arguments:
            counter {dict} -- counter
            sentences {list} -- text content after preprocessing
        """

        for field in self.fields:
            if field.namespace not in self.pretrained_namespace:
                field.count_vocab_items(counter, sentences)

    def index(self, vocab, sentences):
        """This funtion indexes token using vocabulary,
        then update instance

        Arguments:
            vocab {Vocabulary} -- vocabulary
            sentences {list} -- text content after preprocessing
        """

        for field in self.fields:
            field.index(self.instance, vocab, sentences)

    def get_instance(self):
        """This function get instance

        Returns:
            dict -- instance
        """

        return self.instance

    def get_size(self):
        """This funtion gets the size of instance

        Returns:
            int -- instance size
        """

        return len(self.instance[self.fields[0].namespace])
