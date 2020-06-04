import tensorflow as tf
import pandas as pd
from pathlib import Path
from transformers import *
import tensorflow_datasets


def preapre_environment():
    physical_devices = tf.config.list_physical_devices('GPU')
    # print(physical_devices)

    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        # print('Tensorflow sets memory gropth to True')
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass


def prepare_dataframe_dataset(prefix=Path('dataset'), dataset: str = 'train'):
    targets = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult',
               'identity_hate']
    df = pd.read_csv(prefix / f'{dataset}.csv')

    df['class'] = df[targets].max(axis=1)
    df = df.set_index('id')
    df = df.drop(targets, axis=1)

    df.to_csv(prefix / f'{dataset}_processed.csv', index_label='idx')


def prepare_dataframe_unlabeled_set(prefix=Path('dataset'), dataset: str = 'test'):
    targets = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult',
               'identity_hate']
    df_labels = pd.read_csv(prefix / f'{dataset}_labels.csv')
    df_labels['class'] = df_labels[targets].max(axis=1)
    df_labels = df_labels.drop(targets, axis=1)

    df = pd.read_csv(prefix / f'{dataset}.csv')
    df = df.set_index('id').join(df_labels.set_index('id'), on='id')
    df.to_csv(prefix / f'{dataset}_processed.csv', index_label='idx')


def prepare_dataset(prefix=Path('dataset'), dataset: str = 'train'):
    dataframe = pd.read_csv(prefix / f'{dataset}_processed.csv')

    return tf.data.Dataset.from_tensor_slices(
        {
            'idx': tf.cast(dataframe['idx'].values, tf.string),
            'text': dataframe['comment_text'].values,
            'toxicy': dataframe['class'].values
        }
    )


def model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    model = TFBertForSequenceClassification.from_pretrained('bert-base-cased')

    return tokenizer, model


def pipeline(ds, batch_size: int = 1):
    ds = ds.shuffle(1024).batch(batch_size)\
        .prefetch(tf.data.experimental.AUTOTUNE)
    for example in ds.take(1):
        print(example)
        # idx, text, target = example['idx'], example['text'], example['toxicy']
        # print(idx, text, target)


def test():
    data = tensorflow_datasets.load('glue/mrpc')

    for example in data['train'].take(1):
        print(example)


def main():
    preapre_environment()
    # prepare_dataframe_dataset()
    # prepare_dataframe_unlabeled_set()
    train_dataset = prepare_dataset(dataset='train')
    # test_dataset = prepare_dataset(dataset='test')

    pipeline(train_dataset)
    test()


if __name__ == '__main__':
    main()
