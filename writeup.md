# **Semantic Segmentation**


### Semantic segmentation model for specific target tracking

---

### **Project Implementation Overview**

* this project is implemented in tensorflow
* the data provider functionality is implemented in ```segment_net/dataset.py```, using the new dataset feature.
* model training and evaluation is implemented in ```segment_net/train.py```
* different model architectures can be chosen by flag ```FLAGS.name```, the training code in ```segment_net/train.py``` is independent of specific model architecture
* two architectures are implemented in file ```segment_net/fcn_baseline.py``` and ``` segment_net/fcn.py``` the first is a baseline implementation of encoder-decoder architecture, the second is a reproduce of FCN segmentation network with multi-hop skip connection
* new model architecture can be added by register in file ```segment_net/network.py```


### **Project Setup**

Steps of this project are the following:
* download or collect segmentation dataset, preprocessed as images and masks
* download evaluation dataset (sample_evaluation_data)
* download or collect validation dataset, as training dataset
* tensorflow version >= 1.3.1 (to use the dataset features)
* keras version >= 2.0.8 (to run follow-me in simulator)
* create a checkpoint directory: ``` $ mkdir ckpt```


### **Model Training**

training command:

```
$ python -m segment_net --name=fcn --train=./datasets/train/ --valid=./datasets/validation/ --evaluate=./datasets/sample_evaluation_data/ --save=./ckpt/fcn/ --batch_size=32 --crop_size=128 --log=./ckpt/fcn/train.log
```

* meaning of flags are explained in file ```segment_net/flags.py```
* help information can be accessed by ```python -m segment_net --help ```
* ```--train --valid --evaluate``` flags are used to specify training, validation and evaluation dataset, dataset is a directory containing two subdirectories: ```images masks```
* evaluation dataset should be the unzipped ```sample_evaluation_data.zip```
* ```--name``` flag can be used to specify model architecture.check the supported architecture in ```segment_net/network.py```
* ```--save``` flag is used to specify checkpoint directory
* ```--log ``` flag can be used to specify the output file of training and evaluation log
* ```--crop_size``` flag is used to specify the crop size of training example from original image and mask (maximum 256). Larger the crop_size, slower the training speed


### **Model Evaluation**

* once the model is trained, it can be evaluated in notebook ```model_evaluation.ipynb```
* load trained model in ```cell[2]```, ```KerasWrapper``` is a class wrap a keras like model API around tensorflow implementation (to be able to use evaluation functions provided by the project)
* change the ```name='fcn', path='ckpt/fcn'``` in ```cell[2]``` to the intended model and ckpt
* the example included in file ```model_evaluation.ipynb``` is the evaluation of the best performing FCN implementation in this project, achieves ```final_score=0.571723291159```


### **FCN Model Architecture**

* the model architecture is implemented by four encoders and four decoders, in file ```segment_net/fcn.py```
* the encoders are implemented with one separable convolution layer and one pooling layer, each encoder shrink the spatial dimension of the feature map by **2x**
* all the pooling layer's outputs are included as the output of encoding process, the original image is also included as the output of encoding, making a **image feature pyramid**
* the decoder reduce the encoder output pyramid up-down, in each step
the decoder up-sample a small feature map by factor=2 (by a transpose convolution layer), depth concat the filters with the next level feature map, and then aggregate the combined feature by a convolution layer.
* the last output of the decoder's spatial size is the same as the original image, a 1x1 convolution layer is applied to reduce the filters to num_classes (which is 3 in this project), the output feature map is the logits (unormalized log probability of each class) of each pixels

* the graphical illustration of this architecture:

[image_0]: ./docs/misc/fcn_network.png
![alt text][image_0]


### **Encoder-Decoder Architecture Intuition**

The intuition behind encoder-decoder architecture is to build an image feature map that contains simultaneously the high level semantic information and low level geometric information:

  * the encoders build successive features by stacking convolutions and pooling, each layer encodes information at different level. The higher level layers encodes more semantic information and less spatial information.

  * the decoders perform information integration. At each level, one decoder layer integrates a higher level feature map with a lower level feature map into a new feature map therefore distributing the semantic features over spatial features.

  * the feature map before logits layer (one_by_one) encodes semantic information on each pixel

### **One by One Convolutions**

one-by-one convolutions are linear transforms over depth dimension of feature maps (followed by an activation).

mathematically, for a NHWC tensor T of shape ```(a, b, c, d)```:

  * one-by-one convolution is equivalent to:

    ```W.shape = (d, e); b.shape = (e,) ```

    ```conv_one_by_one(T) := reshape(matmul(reshape(T, (a, -1, d)), W) + b, (a, b, c, e))```

* fully-connected layer is equivalent to a full-kernel-size convolution (kernel_size=image_size):

    ```W.shape = (b * c * d, e); b.shape = (e,) ```

    ``` fc(T) := matmul(reshape(T, (a, -1)), W) + b```

the operation is used in this project to reduce the last decoder layer output feature-maps depth to the number of semantic classes (in order to generate logits on each pixel)

another usage is in the separable convolution operator. in separable convolutions decomposes convolution into a spatial transform (on each feature map) and followed by a depth transform (a one-by-one convolution)

### **Training Implementation**

* the training pipeline is implemented in ```segment_net/train.py:227 train_main()```
* the pipeline basically define a set of functions: ```train_fn, validate_fn, score_fn, save_fn, and restore_fn``` using python's closure feature
* dataset is loaded by function ```create_dataset()``` called in each function generator (e.g. ```81: train_network(sess)```)
* the network variables are shared across different application (``` i.e. train, score, validate```) by tensorflow's ```variable_scope``` parameter sharing mechanism
* the score_fn compute the pixel-wise averaged (**across the whole dataset**) mean-iou in three datasets: following, patrol_with_targ, overall evaluation dataset, this score_fn implementation is different from the final score provided by the project **(the mean-iou of each image averaged over images)**. Yet it is a good indicator for the final score


### **Convert To Keras**

* the checkpoint of fcn model can be converted to keras format by:

  ```$ python ckpt_to_keras.py --save=./ckpt/fcn --output=./keras_model```

* the ```--save``` flag points to the directory of tensorflow checkpoint, ```--output``` flag points to the directory of keras_model


### **Parameter Tuning**

* **crop_size:** the random crop size affect the training speed and the quality of the model. Too small crop size will degrade the model quality while too large crop size will increase the epoch time, 128x128 is a good trade-off after several trails (candidates 64x64, 160x160, 256x256(full size))
* **bath_size:** batch size can be in range 16 - 128, too large batch size will increase the time spend on each update step, hence elongate the convergence time. too small batch size is bad in GPU utilization, 32 is a good trade-off
* **epoch_size:** the dataset is relatively small (about 10,000 images), hence the model takes many epochs to converge. I choose a very large pre-set epoch size (500), and manually stop training if the IOU on validation set stops increasing.
* **learning_rate:** I use learning rate by default in tensorflow adam optimization and find it is good enough.


### **hyperparameters**

* training image and mask are randomly cropped as ``` 128x128 ```
* ``` num_encoders=4, num_decoders=4, encoder_filters=64, decoder_filters=64 ```
* all the covolution except the last one is ```3x3```, the last one is ```1x1```, all activations are ```relu```
* all the pooling in encoder is ```size=(2, 2), stride=2```
* no regularization of any kind is applied (for simplicity)
* batch size is chosen as ```batch_size=32```
* batch normalization is **not** applied
* the training stops at ```epoch=395```
* all the training information is recorded in ``` ckpt/fcn/train.log ```

### **DataSet Collection**

6 extra datasets are collected from the simulator
* follow the target hero in a dense crowd
* circle around the hero in high attitude
* follow the hero in a straight line in high attitude
* follow the hero in a straight line in low attitude
* patrol in a alley lane
* patrol in a very long trail

### **Extension to Other Classes**

* the model parameter trained on this task could not be generalized to many more classes.
* the architecture could be reused by collecting training data of other classes and retrain the model.


### **FCN Results**

* as evaluated in ```model_evaluation.ipynb```, the model achives:

    * total iou in evaluation dataset: ```weights = 0.8176795580110497 ```
    * following dataset: ```iou = 0.956149201192 ```
    * with_targ dataset: ```iou = 0.442255065021 ```
    * ``` final_score = 0.571723291159 > 0.4``` (the project requirement)
    * the ckpt is included in this repo

* the model is converted to keras fmt and evaluated in ```keras_evaluation.ipynb ```

* the converted keras model performance (final_score=0.568771257547) is a little lower than the above since the above model predicts hard-class (see ```segment_net/network.py: 92```) while the keras model s softmax

### **Follow Me in Simulator**

* the project has included a modified follower.py:

  * add ```os.environ['CUDA_VISIBLE_DEVICES'] = '' ``` to prevent using gpu in inference
  * load model through ```model_tools.load_keras_model``` (the directory structure of this project is different from the original one)
  * change ``` import tensorflow.contrib.keras  as keras ``` to ``` import keras ```, the ```contrib.keras``` run into cuda error when tested on my machine.

* run follow-me trail: ```$ python follower.py ./keras_model/fcn --pred_viz``` in project directory



### **Future Improvements**

* simultaneously objects tracking and semantic segmentation, to enable the drone to follow any specific target
* apply probabilistic constraints in the output layer (e.g. CRF), to make the pixel predict more consistent
