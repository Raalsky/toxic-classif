import tensorflow as tf
import tensorflow_hub as hub

bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/1", trainable = False)
MAX_LEN = 128


def make_tokenizer(bert_layer):
    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    cased = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = tokenization.FullTokenizer(vocab_file, cased)

    return tokenizer


tokenizer = make_tokenizer(bert_layer)


def preprocess(data, max_seq_length=MAX_LEN, tokenizer=tokenizer):
    ids = []
    masks = []
    segment = []
    for sentence in data:

        tokens = tokenizer.tokenize(sentence)
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[: max_seq_length - 2]

        # Converting tokens to ids
        input_ids = tokenizer.convert_tokens_to_ids(
            ["[CLS]"] + tokens + ["[SEP]"])

        # Input mask
        input_masks = [1] * len(input_ids)

        # padding upto max length
        padding = max_seq_length - len(input_ids)
        input_ids.extend([0] * padding)
        input_masks.extend([0] * padding)
        segment_ids = [0] * max_seq_length

        ids.append(input_ids)
        masks.append(input_masks)
        segment.append(segment_ids)

    return (np.array(ids), np.array(masks), np.array(segment))

train_ids, train_masks, train_segment = preprocess(train_df["comment_text"].values)

model.predict((ids, masks, segment))