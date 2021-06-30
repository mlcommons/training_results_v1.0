# Contents

- [Contents](#contents)
- [BERT Description](#bert-description)
- [Model Architecture](#model-architecture)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
        - [Pre-Training](#pre-training)
    - [Options and Parameters](#options-and-parameters)
        - [Options](#options)
        - [Parameters](#parameters)
    - [Training Process](#training-process)
        - [Distributed Training](#distributed-training)
            - [Running on Ascend](#running-on-ascend-1)

# [BERT Description](#contents)

The BERT network was proposed by Google in 2018. The network has made a breakthrough in the field of NLP. The network uses pre-training to achieve a large network structure without modifying, and only by adding an output layer to achieve multiple text-based tasks in fine-tuning. The backbone code of BERT adopts the Encoder structure of Transformer. The attention mechanism is introduced to enable the output layer to capture high-latitude global semantic information. The pre-training uses denoising and self-encoding tasks, namely MLM(Masked Language Model) and NSP(Next Sentence Prediction). No need to label data, pre-training can be performed on massive text data, and only a small amount of data to fine-tuning downstream tasks to obtain good results. The pre-training plus fune-tuning mode created by BERT is widely adopted by subsequent NLP networks.

[Paper](https://arxiv.org/abs/1810.04805):  Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding]((https://arxiv.org/abs/1810.04805)). arXiv preprint arXiv:1810.04805.

[Paper](https://arxiv.org/abs/1909.00204):  Junqiu Wei, Xiaozhe Ren, Xiaoguang Li, Wenyong Huang, Yi Liao, Yasheng Wang, Jiashu Lin, Xin Jiang, Xiao Chen, Qun Liu. [NEZHA: Neural Contextualized Representation for Chinese Language Understanding](https://arxiv.org/abs/1909.00204). arXiv preprint arXiv:1909.00204.

# [Model Architecture](#contents)

The backbone structure of BERT is transformer. For BERT_base, the transformer contains 12 encoder modules, each module contains one self-attention module and each self-attention module contains one attention module. For BERT_NEZHA, the transformer contains 24 encoder modules, each module contains one self-attention module and each self-attention module contains one attention module. The difference between BERT_base and BERT_NEZHA is that BERT_base uses absolute position encoding to produce position embedding vector and BERT_NEZHA uses relative position encoding.

# [Environment Requirements](#contents)

- Hardware（Ascend/GPU）
    - Prepare hardware environment with Ascend processor.
- Framework
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start pre-training, as follows:

- Prepare dataset
Following the official instruction to get en-wiki dataset in tfrecord format.

- Prepare checkpoint
Download the official checkpoint in tf1 format.
Use scripts in `ms_ckpt_converter` to convert checkpoint to MindSpore format.
```bash
cd ms_ckpt_converter/1.2
bash convert.sh
```

- Running on Ascend locally
Modify arguments in the `ini` config file, e.g. `modelarts/params_128px24.ini`
```bash
# run distributed pre-training example
bash scripts/run_distributed_pretrain_ascend.sh /path/modelarts/params_128px24.ini /path/hccl.json
```

- Running on Ascend with modelarts
Modify arguments in the `py` config file, e.g. `modelarts/params_128px24.py`
```bash
# run on modelarts
cd modelarts
python start_task.py
```

For distributed training on Ascend, an hccl configuration file with JSON format needs to be created in advance.

For distributed training on single machine, [here](https://gitee.com/mindspore/mindspore/tree/master/config/hccl_single_machine_multi_rank.json) is an example hccl.json.

For distributed training among multiple machines, training command should be executed on each machine in a small time interval. Thus, an hccl.json is needed on each machine. [here](https://gitee.com/mindspore/mindspore/tree/master/config/hccl_multi_machine_multi_rank.json) is an example of hccl.json for multi-machine case.

Please follow the instructions in the link below to create an hccl.json file in need:
[https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools).

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```shell
.
└─bert
  ├─README.md
  ├─scripts
    ├─ascend_distributed_launcher
        ├─__init__.py
        ├─hyper_parameter_config.ini          # hyper parameter for distributed pretraining
        ├─get_distribute_pretrain_cmd.py          # script for distributed pretraining
        ├─README.md
    ├─run_classifier.sh                       # shell script for standalone classifier task on ascend or gpu
    ├─run_ner.sh                              # shell script for standalone NER task on ascend or gpu
    ├─run_squad.sh                            # shell script for standalone SQUAD task on ascend or gpu
    ├─run_standalone_pretrain_ascend.sh       # shell script for standalone pretrain on ascend
    ├─run_distributed_pretrain_ascend.sh      # shell script for distributed pretrain on ascend
    ├─run_distributed_pretrain_gpu.sh         # shell script for distributed pretrain on gpu
    └─run_standaloned_pretrain_gpu.sh         # shell script for distributed pretrain on gpu
  ├─src
    ├─__init__.py
    ├─assessment_method.py                    # assessment method for evaluation
    ├─bert_for_finetune.py                    # backbone code of network
    ├─bert_for_pre_training.py                # backbone code of network
    ├─bert_model.py                           # backbone code of network
    ├─finetune_data_preprocess.py             # data preprocessing
    ├─cluner_evaluation.py                    # evaluation for cluner
    ├─config.py                               # parameter configuration for pretraining
    ├─CRF.py                                  # assessment method for clue dataset
    ├─dataset.py                              # data preprocessing
    ├─finetune_eval_config.py                 # parameter configuration for finetuning
    ├─finetune_eval_model.py                  # backbone code of network
    ├─sample_process.py                       # sample processing
    ├─utils.py                                # util function
  ├─modelarts
    ├─start_task.py                           # Start task on modelarts
    ├─params_128px24.py                       # arguments used with modelarts
    ├─params_128px24.ini                      # arguments used locally
    ├─params_256px16.py                       # arguments used with modelarts
    ├─params_256px16.ini                      # arguments used locally
    ├─start_task_copy_dataset.py              # Start task to copy data
    ├─train_cloud_copy_dataset.py             # Modelarts task to copy dataset
  ├─ms_ckpt_converter
    ├─convert.sh                              # Bash scripts to convert checkpoint
    ├─ms2tf_config.py                         # Parameter mapping config
    ├─ms_and_tf_checkpoint_transfer_tools.py  # Converter tools
  ├─pretrain_eval.py                          # train and eval net  
  ├─run_classifier.py                         # finetune and eval net for classifier task
  ├─run_ner.py                                # finetune and eval net for ner task
  ├─run_pretrain.py                           # train net for pretraining phase
  └─run_squad.py                              # finetune and eval net for squad task
```

## [Script Parameters](#contents)

### Pre-Training

```text
usage: run_pretrain.py  [--distribute DISTRIBUTE] [--epoch_size N] [----device_num N] [--device_id N]
                        [--enable_save_ckpt ENABLE_SAVE_CKPT] [--device_target DEVICE_TARGET]
                        [--enable_lossscale ENABLE_LOSSSCALE] [--do_shuffle DO_SHUFFLE]
                        [--enable_data_sink ENABLE_DATA_SINK] [--data_sink_steps N]
                        [--accumulation_steps N]
                        [--allreduce_post_accumulation ALLREDUCE_POST_ACCUMULATION]
                        [--save_checkpoint_path SAVE_CHECKPOINT_PATH]
                        [--load_checkpoint_path LOAD_CHECKPOINT_PATH]
                        [--save_checkpoint_steps N] [--save_checkpoint_num N]
                        [--data_dir DATA_DIR] [--schema_dir SCHEMA_DIR] [train_steps N]

options:
    --device_target                device where the code will be implemented: "Ascend" | "GPU", default is "Ascend"
    --distribute                   pre_training by several devices: "true"(training by more than 1 device) | "false", default is "false"
    --epoch_size                   epoch size: N, default is 1
    --device_num                   number of used devices: N, default is 1
    --device_id                    device id: N, default is 0
    --enable_save_ckpt             enable save checkpoint: "true" | "false", default is "true"
    --enable_lossscale             enable lossscale: "true" | "false", default is "true"
    --do_shuffle                   enable shuffle: "true" | "false", default is "true"
    --enable_data_sink             enable data sink: "true" | "false", default is "true"
    --data_sink_steps              set data sink steps: N, default is 1
    --accumulation_steps           accumulate gradients N times before weight update: N, default is 1
    --allreduce_post_accumulation  allreduce after accumulation of N steps or after each step: "true" | "false", default is "true"
    --save_checkpoint_path         path to save checkpoint files: PATH, default is ""
    --load_checkpoint_path         path to load checkpoint files: PATH, default is ""
    --save_checkpoint_steps        steps for saving checkpoint files: N, default is 1000
    --save_checkpoint_num          number for saving checkpoint files: N, default is 1
    --train_steps                  Training Steps: N, default is -1
    --data_dir                     path to dataset directory: PATH, default is ""
    --schema_dir                   path to schema.json file, PATH, default is ""
```

## Options and Parameters

### Options

```text
config for lossscale and etc.
    bert_network                    version of BERT model: base | nezha, default is base
    batch_size                      batch size of input dataset: N, default is 16
    loss_scale_value                initial value of loss scale: N, default is 2^32
    scale_factor                    factor used to update loss scale: N, default is 2
    scale_window                    steps for once updatation of loss scale: N, default is 1000
    optimizer                       optimizer used in the network: AdamWerigtDecayDynamicLR | Lamb | Momentum, default is "Lamb"
```

### Parameters

```text
Parameters for dataset and network (Pre-Training/Fine-Tuning/Evaluation):
    seq_length                      length of input sequence: N, default is 128
    vocab_size                      size of each embedding vector: N, must be consistent with the dataset you use. Default is 21128.
                                    Usually, we use 21128 for CN vocabs and 30522 for EN vocabs according to the origin paper.
    hidden_size                     size of bert encoder layers: N, default is 768
    num_hidden_layers               number of hidden layers: N, default is 12
    num_attention_heads             number of attention heads: N, default is 12
    intermediate_size               size of intermediate layer: N, default is 3072
    hidden_act                      activation function used: ACTIVATION, default is "gelu"
    hidden_dropout_prob             dropout probability for BertOutput: Q, default is 0.1
    attention_probs_dropout_prob    dropout probability for BertAttention: Q, default is 0.1
    max_position_embeddings         maximum length of sequences: N, default is 512
    type_vocab_size                 size of token type vocab: N, default is 16
    initializer_range               initialization value of TruncatedNormal: Q, default is 0.02
    use_relative_positions          use relative positions or not: True | False, default is False
    dtype                           data type of input: mstype.float16 | mstype.float32, default is mstype.float32
    compute_type                    compute type in BertTransformer: mstype.float16 | mstype.float32, default is mstype.float16

Parameters for optimizer:
    AdamWeightDecay:
    decay_steps                     steps of the learning rate decay: N
    learning_rate                   value of learning rate: Q
    end_learning_rate               value of end learning rate: Q, must be positive
    power                           power: Q
    warmup_steps                    steps of the learning rate warm up: N
    weight_decay                    weight decay: Q
    eps                             term added to the denominator to improve numerical stability: Q

    Lamb:
    decay_steps                     steps of the learning rate decay: N
    learning_rate                   value of learning rate: Q
    end_learning_rate               value of end learning rate: Q
    power                           power: Q
    warmup_steps                    steps of the learning rate warm up: N
    weight_decay                    weight decay: Q

    Momentum:
    learning_rate                   value of learning rate: Q
    momentum                        momentum for the moving average: Q
```
