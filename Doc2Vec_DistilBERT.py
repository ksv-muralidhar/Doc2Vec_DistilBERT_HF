import tensorflow as tf
from transformers import DistilBertTokenizerFast, TFDistilBertModel
import warnings
warnings.filterwarnings('ignore')
from transformers import logging
logging.set_verbosity_error()


class DistilBertDoc2Vec:
    '''
    Returns Doc2Vec embeddings of documents in a corpus.
    
    Params:
    n_hidden_states: int
     Number of hidden states of DistilBERT to consider to compute Doc2Vec
    
    n_tokens: int
     Max tokens in a document
     
    use_only_cls_token: bool
     Boolean specifying to use only [CLS] token embedding or compute mean of all the tokens
     
    return_tf_tensor: bool
     Boolean specifying if Doc2Vec embeddings must be returned as tf tensor or numpy array.
    '''
    def __init__(self, n_hidden_states: int=6, n_tokens: int=512, use_only_cls_token: bool=False,
                return_tf_tensor: bool=True):
        self.n_hidden_states = n_hidden_states
        self.n_tokens = n_tokens
        self.use_only_cls_token = use_only_cls_token
        self.return_tf_tensor = return_tf_tensor
        self.checkpoint = 'distilbert-base-uncased'
    
    @staticmethod
    def __get_valid_input(value: int, bounds: list[int]):
        '''
        Forces the value to be within the bounds.
        
        Params:
        value: int
         Value to be checked for validity.
         
        bounds: list[int]
         Bounds within which the value must be present
        '''
        value = bounds[1] if value > bounds[1] else value
        value = bounds[0] if value < bounds[0] else value
        return value
    
    def __get_tokens(self, corpus: list[str]):
        '''
        Tokenize the corpus.
        '''
        tokenizer = DistilBertTokenizerFast.from_pretrained(self.checkpoint)
        tokens = tokenizer(corpus, 
                           max_length=self.n_tokens, 
                           padding="max_length", 
                           truncation=True, 
                           return_attention_mask=True, 
                           return_tensors='tf')
        return tokens
    
    
    def __get_model_embeddings(self, corpus: list[str]):
        '''
        Return embeddings of the corpus
        '''
        model = TFDistilBertModel.from_pretrained(self.checkpoint, output_hidden_states=True)
        tokens = self.__get_tokens(corpus)
        model_embeddings = model(input_ids=tokens['input_ids'], attention_mask=tokens['attention_mask'])
        return model_embeddings
        
    def get_doc2vec(self, corpus: list[str]):
        '''
        Get Doc2Vec embeddings for the corpus.
        
        Params:
        corpus: list[str]
         List of documents.
         
        Returns:
        doc2vec
         Doc2Vec embeddings of the corpus
        '''
        n_hidden_states = self.__get_valid_input(self.n_hidden_states, [1,6])
        n_tokens = self.__get_valid_input(self.n_tokens, [1,512])
        model_embeddings = self.__get_model_embeddings(corpus)
        hidden_states = model_embeddings[1][-n_hidden_states:]
        hidden_states_tf = tf.convert_to_tensor(hidden_states)
        if self.use_only_cls_token:
            hidden_states_doc2vec = hidden_states_tf[:,:,0,:]
        else:
            hidden_states_doc2vec = tf.map_fn(lambda x: tf.math.reduce_mean(x[:,1:,:], axis=1), hidden_states_tf)
        
        doc2vec = tf.math.reduce_mean(hidden_states_doc2vec, axis=0)
        
        if self.return_tf_tensor:
            return doc2vec
        
        return doc2vec.numpy()
