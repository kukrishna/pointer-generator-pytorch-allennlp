PointerGenerator network implementation in AllenNLP
---


This repo contains an implementation of the pointer-generator network from [See et al (2017)](https://arxiv.org/abs/1704.04368) 
in PyTorch wrapped in the AllenNLP framework.

#### Running pretrained model

We are releasing the pretrained model parameters from See et al converted into
PyTorch's format. You can download the pretrained parameter file from [here](https://drive.google.com/drive/folders/1vU6-npbiPjw4ZUuy1LXzaN6XhETWrg74?usp=sharing) 

To run the pretrained model, use the command


`allennlp predict pretrained_model_skeleton.tar.gz --weights-file <weights-file> --include-package pointergen --cuda-device <gpu-id> --predictor beamsearch <test-file>`

where `weights-file` is the PyTorch pretrained weights file downloaded from the link above. A sample
`test-file` has been provided to illustrate the input jsonl format. The output has been tested to
match the output of the tensorflow model parameters released by the authors. 


#### Training your own model

The configuration for training a model is given in the `sample_experiment.jsonnet` file. You have to provide paths
 to the training and validation jsonl files in it. You can also customize the model size
  and optimizer parameters via this file. To train a model, run the command

`allennlp train sample_experiment.jsonnet --include-package pointergen -s <serialization-directory>`


Currently the model does not support coverage. Support for coverage will be added soon.


allennlp predict pretrained_model_skeleton.tar.gz --weights-file see_etal_weights_without_coverage.th --include-package pointergen --cuda-device 0 --predictor beamsearch sample_datafile.jsonl