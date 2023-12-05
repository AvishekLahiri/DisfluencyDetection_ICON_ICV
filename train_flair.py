from flair.data import Corpus
from flair.datasets import ColumnCorpus

# define columns
columns = {0: 'text', 1: 'disf'}

# this is the folder in which train, test and dev files reside
data_folder = '/path/to/data/folder/'

# Please note that the train.tsv file contains training data from both training files 
corpus: Corpus = ColumnCorpus(data_folder, columns,
                              train_file='train.tsv',
                              dev_file='dev.tsv')


from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer


# The label which we want to predict.
label_type = 'disf'

# Make the label dictionary from the corpus
label_dict = corpus.make_label_dictionary(label_type=label_type, add_unk=False)
print(label_dict)

# Initialize fine-tuneable transformer embeddings WITH document context
embeddings = TransformerWordEmbeddings(model='google/muril-base-cased',
                                       layers="all",
                                       subtoken_pooling="first",
                                       fine_tune=True,
                                       use_context=True,
                                       model_max_length=512
                                       )

# Initialize bare-bones sequence tagger (with CRF but no RNN and no reprojection)
tagger = SequenceTagger(hidden_size=256,
                        embeddings=embeddings,
                        tag_dictionary=label_dict,
                        tag_type='disf',
                        use_crf=True,
                        use_rnn=False,
                        reproject_embeddings=False,
                        dropout = 0.2,
                        tag_format="BIO"
                        )

# Initialize trainer
trainer = ModelTrainer(tagger, corpus)

# Run fine-tuning
trainer.fine_tune('disf_model_Bengali',
                  learning_rate=0.1,
                  mini_batch_size=4,
                  max_epochs=15)