{
    "dataset_reader": {
        "lazy": false,
        "type": "cnndmail_dataset_reader",
        "lowercase_tokens": true,
    },
    "validation_dataset_reader": {
        "lazy": false,
        "type": "cnndmail_dataset_reader",
        "lowercase_tokens": true,
    },
    "datasets_for_vocab_creation": ["train"],
    "train_data_path": "sample_datafile.jsonl",        // path to training jsonl file
    "validation_data_path": "sample_datafile.jsonl",   // path to validation jsonl file
    "vocabulary": {
        "extend": true,
        "directory_path": "bootstrapped_vocabulary",
          "max_vocab_size": 49998     // note that this is the max number of words to add on top of the words already present in bootstrapped vocab after unk ie. (start, end)
    },
    "model": {
        "type": "pointer_generator",
        "hidden_size": 64,
        "emb_size": 32,
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys": [["source_tokens", "num_tokens"]],
        "batch_size": 16
    },
    "trainer": {
        "optimizer": {
            "type": "adam",
            "lr": 0.001
        },
        "validation_metric": "-loss",
        "num_serialized_models_to_keep": 1,
        "num_epochs": 100,
        "grad_norm": 10.0,
        "patience": 50,
        "cuda_device": 0
    }
}
