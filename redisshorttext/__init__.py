from converter import *
from classifier import *
import redis

class GroceryException(Exception):
    pass


class GroceryNotTrainException(GroceryException):
    def __init__(self):
        self.message = 'Text model has not been trained.'


class Grocery(object):
    def __init__(self,name=None, custom_tokenize=None,rhost='localhost',port=6379):
        self.name = name
        if custom_tokenize is not None:
            if not hasattr(custom_tokenize, '__call__'):
                raise GroceryException('Tokenize func must be callable.')
        self.custom_tokenize = custom_tokenize
        self.classifier = None
        self.train_svm = "%s_train.svm" % name
        self.redishost = rhost
        self.redisport = port

    @property
    def tmodel(self):

        return train(data_file_name=self.train_svm, learner_opts='',\
                liblinear_opts='-s 4',redishost=self.redishost,redisport=self.redisport)

    @property
    def text_converter(self):

        return GroceryTextConverter(redis=self.myredis,\
                custom_tokenize=self.custom_tokenize)

    @property
    def myredis(self):
        
        return redis.Redis(host=self.redishost,port=self.redisport)

    @property
    def model(self):

        return GroceryTextModel(redis=self.myredis,text_converter=self.text_converter,\
                model=self.tmodel,custom_tokenize=self.custom_tokenize)


    def train(self, row):

        self.text_converter.convert_text(row, output=self.train_svm)


    def predict(self, single_text):

        return self.model.predict_text(single_text).predicted_y


