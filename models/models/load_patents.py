# _*_ coding: utf-8 _*_

import os
import sys
import torch
from torch.nn import functional as F
import numpy as np
#from torchtext import datasets
from torchtext.vocab import Vectors, GloVe

from torchtext.data import Field, LabelField, Dataset, Example, BucketIterator
import pandas as pd
import json

class DataFrameDataset(Dataset):
    """Class for using pandas DataFrames as a datasource"""
    def __init__(self, examples, fields, filter_pred=None):
        """
        Create a dataset from a pandas dataframe of examples and Fields
        Arguments:
         examples pd.DataFrame: DataFrame of examples
         fields {str: Field}: The Fields to use in this tuple. The
             string is a field name, and the Field is the associated field.
         filter_pred (callable or None): use only exanples for which
             filter_pred(example) is true, or use all examples if None.
             Default is None
        """
        self.examples = examples.apply(SeriesExample.fromSeries, args=(fields,), axis=1).tolist()
        if filter_pred is not None:
            self.examples = filter(filter_pred, self.examples)
        self.fields = dict(fields)
        # Unpack field tuples
        for n, f in list(self.fields.items()):
            if isinstance(n, tuple):
                self.fields.update(zip(n, f))
                del self.fields[n]

class SeriesExample(Example):
    """Class to convert a pandas Series to an Example"""

    @classmethod
    def fromSeries(cls, data, fields):
        return cls.fromdict(data.to_dict(), fields)

    @classmethod
    def fromdict(cls, data, fields):
        ex = cls()

        for key, field in fields.items():
            if key not in data:
                raise ValueError("Specified key {} was not found in "
                "the input data".format(key))
            if field is not None:
                setattr(ex, key, field.preprocess(data[key]))
            else:
                setattr(ex, key, data[key])
        return ex
        # for key, tuple in fields.items():
        #     (name, field) = tuple
        #     if key not in data:
        #         raise ValueError("Specified key {} was not found in "
        #         "the input data".format(key))
        #     if field is not None:
        #         setattr(ex, name, field.preprocess(data[key]))
        #     else:
        #         setattr(ex, name, data[key])
        # return ex

def load_dataset(batch_size, test_sen=None):

    office_actions = pd.read_csv('/mnt/data/training-patent-data4144f61d-a15b-421e-9346-659741ee1c22/office_actions.csv', usecols=['app_id', 'ifw_number', 'rejection_102', 'rejection_103'], nrows=100000)

    abstractList = []
    idList = []
    rejectionColumn = []
    for num in range(10000):

        app_id = str(office_actions.app_id[num])
        filename = "/mnt/data/training-patent-data4144f61d-a15b-421e-9346-659741ee1c22/json_files_1/oa_"+app_id+".json"

        try:
            jfile = open(filename, 'r')
        except FileNotFoundError:
            print("File Not Found")
            continue

        parsed_json = json.load(jfile)
        jfile.close()

        try:
            abstractList.append(parsed_json[0]['abstract_full'])
            idList.append(parsed_json[0]['application_number'])
        except IndexError:
            print("WARNING: file "+filename+" is empty!\n")
            continue

        n = int(office_actions.rejection_102[num])
        o = int(office_actions.rejection_103[num])

        if n == 0 and o == 0:
            rejType = 0 #neither
        elif n == 0 and o == 1:
            rejType = 1 #obvious
        elif n == 1 and o == 0:
            rejType = 0 #novelty
        elif n == 1 and o == 1:
            rejType = 1 #both
        else:
            print("Office action error:", sys.exc_info()[0])
            raise

        rejectionColumn.append(rejType)

    all_data = {'text': abstractList, 'label': rejectionColumn}
    df = pd.DataFrame(all_data, index = idList)

    tokenize = lambda x: x.split()
    TEXT = Field(sequential=True, tokenize=tokenize, lower=True, include_lengths=True, batch_first=True, fix_length=200)
    LABEL = LabelField(sequential=False)
    #fields={'Abstract': ('text', TEXT), 'RejectionType': ('labels', LABEL)}
    fields={'text': TEXT, 'label': LABEL}


    ds = DataFrameDataset(df, fields)

    TEXT.build_vocab(ds, vectors=GloVe(name='6B', dim=300))
    LABEL.build_vocab(ds)

    train_data, test_data = ds.split()
    train_data, valid_data = train_data.split() # Further splitting of training_data to create new training_data & validation_data

    word_embeddings = TEXT.vocab.vectors
    print ("Length of Text Vocabulary: " + str(len(TEXT.vocab)))
    print ("Vector size of Text Vocabulary: ", TEXT.vocab.vectors.size())
    print ("Label Length: " + str(len(LABEL.vocab)))

    train_iter, valid_iter, test_iter = BucketIterator.splits((train_data, valid_data, test_data), batch_size=batch_size, sort_key=lambda x: len(x.text), repeat=False, shuffle=True)

    vocab_size = len(TEXT.vocab)

    return TEXT, vocab_size, word_embeddings, train_iter, valid_iter, test_iter
