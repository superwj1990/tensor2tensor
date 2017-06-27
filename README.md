# T2T: Tensor2Tensor Transformers<br>
T2T是一个模块化和可扩展的库和二进制文件，用于基于TensorFlow进行的有监督学习和为序列任务提供支持。它由谷歌大脑团队得工程师和研究员进行维护和使用。你可以在最近的《Google Research Blog post introducing it》一文中获得更多关于Tensor2Tensor的消息。<br>

我们渴望你加入到T2T的拓展中来，你可以尽情在github上提问或者发送一个pull request来添加你的数据库或者模型。可以参阅contribution doc来进一步了解T2T的细节及存在的问题。同时也可以在Gitter上与我们及其他使用者交流。<br>

**目录**<br>
* Walkthrough
*	Installation
*	Features
*	T2T Overview
   * Datasets
   * Problems and Modalities
   * Models
   * Hyperparameter Sets
  * Trainer
    *	Adding your own components
*	Adding a dataset
--------------------------------------------------------------------------------
## Walkthrough<br>
这是在WMT数据上训练一个英语-德语翻译模型的walkthrough，采用《Attention Is All You Need》中的Transformer模型。<br>
```
pip install tensor2tensor

# See what problems, models, and hyperparameter sets are available.
# You can easily swap between them (and add new ones).
t2t-trainer --registry_help

PROBLEM=wmt_ende_tokens_32k
MODEL=transformer
HPARAMS=transformer_base

DATA_DIR=$HOME/t2t_data
TMP_DIR=/tmp/t2t_datagen
TRAIN_DIR=$HOME/t2t_train/$PROBLEM/$MODEL-$HPARAMS

mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR

# Generate data
t2t-datagen \
  --data_dir=$DATA_DIR \
  --tmp_dir=$TMP_DIR \
  --num_shards=100 \
  --problem=$PROBLEM

mv $TMP_DIR/tokens.vocab.32768 $DATA_DIR

# Train
# *  If you run out of memory, add --hparams='batch_size=2048' or even 1024.
t2t-trainer \
  --data_dir=$DATA_DIR \
  --problems=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR

# Decode

DECODE_FILE=$DATA_DIR/decode_this.txt
echo "Hello world" >> $DECODE_FILE
echo "Goodbye world" >> $DECODE_FILE

BEAM_SIZE=4
ALPHA=0.6

t2t-trainer \
  --data_dir=$DATA_DIR \
  --problems=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR \
  --train_steps=0 \
  --eval_steps=0 \
  --decode_beam_size=$BEAM_SIZE \
  --decode_alpha=$ALPHA \
  --decode_from_file=$DECODE_FILE

cat $DECODE_FILE.$MODEL.$HPARAMS.beam$BEAM_SIZE.alpha$ALPHA.decodes
```
--------------------------------------------------------------------------------
## 安装<br>
```
# Assumes tensorflow or tensorflow-gpu installed
pip install tensor2tensor

# Installs with tensorflow-gpu requirement
pip install tensor2tensor[tensorflow_gpu]

# Installs with tensorflow (cpu) requirement
pip install tensor2tensor[tensorflow]
```

二进制文件：<br>
```
# Data generator
t2t-datagen

# Trainer
t2t-trainer --registry_help
```

库的使用方法：<br>
```
python -c "from tensor2tensor.models.transformer import Transformer"
```
--------------------------------------------------------------------------------
## 特点<br>
*	内嵌了许多state of the art和baseline模型，且新的模型可以很容易添加（open an issue or pull request!）<br>
*	拥有多种数据集模式-文本，音频，图像-可用于生成和使用，且很容易添加新内容（pen an issue or pull request for public datasets!）<br>
*	模型可以被用于任何一个数据库和输入模式（甚至是多重形式）；所有modality-specific的处理（如：embedding lookups for text tokens）都可以用`Modality`对象实现，该对象在数据集/任务规范中的每个特征中进行指定。<br>
*	支持multi-GPU机器及同步（1 master, many workers）和异步（independent workers synchronizing through a parameter server）分布式训练。<br>
*	可以轻松通过的数据生成脚本`t2t-datagen`和训练脚本`t2t-trainer`的命令行标志进行数据集和模型之间的互换。<br>
--------------------------------------------------------------------------------
## T2T 概述<br>
### Datasets<br>
数据集通过`tensorflow.Example`的协议缓冲在`TFRecord`文件中进行标准化。所有的数据集通过data generator进行注册和生成，许多常用的序列数据集都已经可以用生成和使用。<br>

### Problems and Modalities<br>
**Problems**用于数据集和任务的训练期间的超参数的定义，如果可以的话，其主要通过设定输入和输出的**Modalities**（如：symbol, image, audio, label）及vocabularies实现。所有的problems都在problem_hparams.py中定义。**Modalities**在modality.py中定义，抽象出输入和输出数据类型，使得**models**可以处理modality-independent tensors。<br>

### Models<br>
`T2Tmodel`定义了核心的tensor-to-tensor转换，独立于input/output模态或任务。Models采用dense tensors及生成dense tensors，然后根据任务在最后一步通过**modality**将它们进行转换（如：fed through a final linear transform to produce logits for a softmax over classes）。所有的模型在models.py中引入，继承于`T2TModel`-在t2t_model.py中定义。<br>
*	通过@registry.register_model注册<br>

### Hyperparameter Sets<br>
**Hyperparameter Sets**在@registry.register_hparams的代码中定义和注册，且在tf.contrib.training.HParams对象中进行编码。`HParams`可用于problem规范和model。common_hparams.py中定义了一个基本的hyperparameters set。hyperparameter set函数可以构成其他hyperparameter set函数。<br>

### Trainer<br>
**Trainer**二进制文件是训练，评价和推断的主要入口。用户可以通过`--model`，`--problems`和`--hparams_set`标记，轻松地在问题、模型和超参数集之间切换。特定的hyperparameters可以通过`--hparams`标记覆写。`--schedule`和相关标记控制本地和分布式的训练/评价（distributed training documentation）。<br>
--------------------------------------------------------------------------------
## 添加你的components<br>
T2T的components使用一个central registration mechanism进行注册，这使得我们可以很容易添加新的component，且可以通过command-line标记来在它们之间轻松地进行切换。你可以通过指定`t2t-trainer`中的`--t2t_usr_dir`标记来添加你自己的components，而不用编辑T2T的基本代码。<br>

你现在可以对 models，hyperparameter sets和modalities进行这些操作。如果你的component对其他人有帮助，你可以提交一个pull request。<br>

这里是一个新的超参数集的示例：
```
# In ~/usr/t2t_usr/my_registrations.py

from tensor2tensor.models import transformer
from tensor2tensor.utils import registry

@registry.register_hparams
def transformer_my_very_own_hparams_set():
  hparams = transformer.transformer_base()
  hparams.hidden_size = 1024
  ...
```

```
# In ~/usr/t2t_usr/__init__.py
import my_registrations
```

```
t2t-trainer --t2t_usr_dir=~/usr/t2t_usr --registry_help
```

你可以在注册的HParams下看到你的`transformer_my_very_own_hparams_set`，你可以直接通过`--hparams_set`标记在命令行直接使用它。<br>

## 添加一个dataset<br>
参见data generators README。<br>
--------------------------------------------------------------------------------
注释：这不是谷歌的官方资料。<br>


