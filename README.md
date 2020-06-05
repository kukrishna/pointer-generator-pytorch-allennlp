PointerGenerator network implementation in AllenNLP
---


This repo contains an implementation of the pointer-generator network from [See et al (2017)](https://arxiv.org/abs/1704.04368) 
in PyTorch wrapped in the AllenNLP framework.

#### Running pretrained model

We are releasing the pretrained model parameters from See et al converted into
PyTorch's format. You can download the pretrained parameter file from [here](https://drive.google.com/drive/folders/1vU6-npbiPjw4ZUuy1LXzaN6XhETWrg74?usp=sharing) 

To run the pretrained model without coverage, use the command

`allennlp predict pretrained_model_skeleton.tar.gz --weights-file see_etal_weights_without_coverage.th --include-package pointergen --cuda-device <gpu-id> --predictor beamsearch <test-file>`

where `weights-file` is the PyTorch pretrained weights file downloaded from the link above. A sample
`test-file` has been provided to illustrate the input jsonl format.


For the model with coverage, use

`allennlp predict pretrained_model_skeleton_withcoverage.tar.gz --weights-file see_etal_weights_with_coverage.th --include-package pointergen --cuda-device <gpu-id> --predictor beamsearch <test-file>`



#### Training your own model

The configuration for training a model is given in the `sample_experiment.jsonnet` file. You have to provide paths
 to the training and validation jsonl files in it. You can also customize the model size
  and optimizer parameters via this file. To train a model, run the command

`allennlp train sample_experiment.jsonnet --include-package pointergen -s <serialization-directory>`

The above command will train the model without the coverage loss. To further train the model with coverage loss, follow the next steps.
Suppose the serialization directory of the model trained without coverage is `precoverage`. Then run


`allennlp train precoverage/config.json --include-package pointergen -s postcoverage -f  --overrides '{"model":{"type":"pointer_generator_withcoverage"}, "trainer":{"num_epochs":1}, "vocabulary": {"extend": true, "directory_path": "precoverage/vocabulary", "max_vocab_size":1}}'`

We will discuss why this command looks so ugly later. After running this command, you would be asked if you want to load weigths from a non-coverage model. Enter yes and in the next prompt enter the path to the weights (`precoverage/best.th` in this case)

To make predictions on a test file, the usual command is used.

`allennlp predict postcoverage/model.tar.gz --include-package pointergen --cuda-device 0 --predictor beamsearch sample_datafile.jsonl`

You would see the same prompt, just reply no to it and the program would continue as usual.


#### What do the entries in overrides flag do?

We prefer to override the settings of the precoverage config.json file rather than creating a new one so that it is ensured that the other parameters of the experiment remain the same (eg. hidden size of LSTM layers). The overrides change the model type to `pointer_generator_coverage` and the number of epochs can be set as required for finetuning. Since we will be loading pretrained weights, we must make sure that the vocabulary of the new model is same as the one that the old one had. This might easily not hold true if for example, you finetune on a slightly different training dataset or a subset of it. So to enforce it we give the path of the vocabulary of the old precoverage model. Ideally the `extend` flag should have been turned to false to disallow further additions to the vocabulary. But then allennlp complains that it was passed an extra parameter `max_vocab_size` which was picked up from the `precoverage/config.json` file. So we set it `extend` to true but pass a very small value of `max_vocab_size` through the override which is certainly even smaller than the loaded vocabulary and hence further additions do not happen to it. Why didn't we set `max_vocab_size` to 0? Because it is interpreted by allennlp as a directive to not place any bounds on the vocabulary size. So we set it to 1 which still has the effect that we want.

