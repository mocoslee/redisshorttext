import uuid
import os
import shutil

from converter import GroceryTextConverter
from .learner import *


class GroceryTextModel(object):
    def __init__(self,redis=None, text_converter=None, model=None,custom_tokenize=None):
        if isinstance(text_converter, GroceryTextConverter):
            self.text_converter = text_converter
        self.custom_tokenize = custom_tokenize
        self.svm_model = model
        self._hashcode = str(uuid.uuid4())
        self.redis = redis

    def __str__(self):
        return 'TextModel instance ({0}, {1})'.format(self.text_converter, self.svm_model)

    def get_labels(self):
        return [self.text_converter.get_class_name(k) for k in self.svm_model.get_labels()]



    def predict_text(self, text):
        if self.svm_model is None:
            raise Exception('This model is not usable because svm model is not given')
        # process unicode type
        if isinstance(text, unicode):
            text = text.encode('utf-8')
        if not isinstance(text, str):
            raise TypeError('The argument should be plain text')
        text = self.text_converter.to_svm(text)
        print "=========1"
        y, dec = predict_one(text, self.svm_model)
        print "=========2"
        y = self.text_converter.get_class_name(int(y))
        print "=========3"
        labels = [self.text_converter.get_class_name(k) for k in
                  self.svm_model.label[:self.svm_model.nr_class]]
        print "=========4"
        return GroceryPredictResult(predicted_y=y, dec_values=dec[:self.svm_model.nr_class], labels=labels)


class GroceryPredictResult(object):
    def __init__(self, predicted_y=None, dec_values=None, labels=None):
        self.predicted_y = predicted_y
        self.dec_values = dec_values
        self.labels = labels
