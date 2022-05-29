import wandb
import optuna
import torch.optim as optim


from flair.data import Corpus
from flair.datasets import CSVClassificationCorpus
from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings, DocumentRNNEmbeddings
from flair.models.text_regression_model import TextRegressor
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from IPython.display import clear_output
from pytorch_lightning.loggers.wandb import WandbLogger


from _lib.settings import DATA_FLAIR_CORPUS_DIR, DATA_W2V_KEYED_VECTORS_DIR, DATA_FLAIR_TESTS_DIR


class DocumentEmbeddingType:
    POOL='pool'
    RNN='rnn'


def load_corpus(path, label_type):
    corpus: Corpus = CSVClassificationCorpus(path, 
                                        {0: "text", 1: "label_topic"}, 
                                        skip_header=True, 
                                        delimiter=';', 
                                        label_type=label_type
    )

    label_dict = corpus.make_label_dictionary(label_type=label_type)

    return corpus, label_dict


def init_document_embedding(**kwargs):
    if kwargs['embedding_type'] == DocumentEmbeddingType.POOL:
        document_embeddings = DocumentPoolEmbeddings([kwargs['glove_embedding']], 
                                                        fine_tune_mode=kwargs['fine_tune_mode'],
                                                        pooling=kwargs['pooling'])

        embed_params = {
            'fine_tune_mode': kwargs['fine_tune_mode'],
            'pooling': kwargs['pooling']
        }

    elif kwargs['embedding_type'] == DocumentEmbeddingType.RNN:
        document_embeddings = DocumentRNNEmbeddings([kwargs['glove_embedding']], 
                                                hidden_size=kwargs['hidden_size'],
                                                rnn_layers=kwargs['rnn_layers'],
                                                bidirectional=kwargs['bidirectional'],
                                                dropout=kwargs['dropout'],
                                                rnn_type=kwargs['rnn_type'])

        embed_params = {
            'hidden_size': kwargs['hidden_size'],
            'rnn_layers': kwargs['rnn_layers'],
            'bidirectional': kwargs['bidirectional'],
            'dropout': kwargs['dropout'],
            'rnn_type': kwargs['rnn_type'],
        }
    
    embed_params['embedding_type'] = kwargs['embedding_type']

    return document_embeddings, embed_params



def train_model(**kwargs):
    logger = WandbLogger()

    run = wandb.init(
        project=kwargs['project_name'], 
        config=kwargs['wandb_config'], 
        entity='bjuggler'
    )

    model = TextClassifier(
        kwargs['document_embeddings'], 
        label_type=kwargs['label_type'], 
        label_dictionary=kwargs['label_dict']
    )

    trainer = ModelTrainer(model, kwargs['corpus'])

    result = trainer.train(f"resources/{kwargs['project_name']}",
                    max_epochs=kwargs['max_epochs'],
                    learning_rate=kwargs['learning_rate'],
                    mini_batch_size=kwargs['mini_batch_size'],
                    optimizer=getattr(optim, kwargs['optimizer']),
                    anneal_factor=kwargs['anneal_factor'],
                    warmup_fraction=kwargs['warmup_fraction'],
                    monitor_test=True,
                    embeddings_storage_mode=None,
                    save_final_model=False,
                    param_selection_mode=kwargs['param_selection_mode']
                    )

    run.finish()

    return result, trainer, model


def test(corpus_dir, vectors_fname, embedding_type, max_epochs=15, n_trials=50):
    def objective(trial):
        trainer__optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'Adagrad', 'SGD']) 
        trainer__learning_rate = trial.suggest_loguniform('lr', 1e-5, 1e-0) 
        trainer__min_batch_size = trial.suggest_int('min_batch_size', 32, 256, step=32) 
        trainer__anneal_factor = trial.suggest_float('anneal_factor', 0.1, 0.6, step=0.1)
        trainer__warmup_fraction = trial.suggest_loguniform('warmup_fraction', 0.01, 0.7)

        document_embeddings, embed_params = init_document_embedding(
                embedding_type=embedding_type,
                glove_embedding=glove_embedding,
                fine_tune_mode=trial.suggest_categorical('fine_tune_mode', ['none', 'linear', 'nonlinear']),
                pooling=trial.suggest_categorical('pooling', ['mean', 'max', 'min']),
                hidden_size=trial.suggest_int('hidden_size', 1, 400),
                rnn_layers=trial.suggest_int('rnn_layers', 1, 5),
                bidirectional=trial.suggest_categorical('bidirectional', [False, True]),
                dropout=trial.suggest_uniform('dropout', 0.1, 0.7),
                rnn_type=trial.suggest_categorical('rnn_type', ['GRU', 'LSTM']) 
        )

        wandb_config = {
            'max_epochs': max_epochs,
            'learning_rate': trainer__learning_rate,
            'mini_batch_size': trainer__min_batch_size,
            'optimizer': trainer__optimizer_name,
            'anneal_factor': trainer__anneal_factor,
            'warmup_fraction': trainer__warmup_fraction
        }

        wandb_config.update(embed_params)

        result, trainer, model = train_model(
            project_name=project_name,
            wandb_config=wandb_config,
            document_embeddings=document_embeddings,
            label_type=label_type,
            label_dict=label_dict,
            corpus=corpus,
            max_epochs=max_epochs,
            learning_rate=trainer__learning_rate,
            mini_batch_size=trainer__min_batch_size,
            optimizer=trainer__optimizer_name,
            anneal_factor=trainer__anneal_factor,
            warmup_fraction=trainer__warmup_fraction,
            param_selection_mode=True
        )

        clear_output(wait=True)

        return result['test_score']

    project_name = f"{embedding_type}_{vectors_fname}_{corpus_dir}"

    label_type = 'mylable'
    corpus, label_dict = load_corpus(f"{DATA_FLAIR_CORPUS_DIR}/{corpus_dir}", label_type)
    glove_embedding = WordEmbeddings(f"{DATA_W2V_KEYED_VECTORS_DIR}/{vectors_fname}.kv")

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    print('params:', study.best_params)
    print('value:', study.best_value)

    study_df = study.trials_dataframe()
    study_df.to_csv(f"{DATA_FLAIR_TESTS_DIR}/{project_name}.csv")