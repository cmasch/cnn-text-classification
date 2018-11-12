# Text classification with Convolution Neural Networks (CNN)
This is a project to classify text documents / sentences with CNNs. You can find a great introduction in a similar approach on a blog entry of [Denny Britz](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/) and [Keras](https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html). My approach is quit similar to the one of Denny and the original paper of Yoon Kim [1]. You can find the implementation of Yoon Kim on [GitHub](https://github.com/yoonkim/CNN_sentence) as well.

## Evaluation
For evaluation I used different datasets that are freely available. They differ in their size of amount and the content length. What all have in common is that they have two classes to predict (positive / negative). I would like to show how CNN performs on ~10000 up to ~200000 documents with modify only a few paramters.

I used the following sets for evaluation:
- [sentence polarity dataset v1.0](http://www.cs.cornell.edu/people/pabo/movie-review-data/)<br>
The polarity dataset v1.0 has 10662 sentences. It's quit similiar to traditional sentiment analysis of tweets because of the content length. I just splitted the data in train / validation (90% / 10%).
- [IMDB moview review](http://ai.stanford.edu/~amaas/data/sentiment/)<br>
IMDB moview review has 25000 train and 25000 test documents. I splitted the trainset into train / validation (80% / 20%) and used the testset for a final test.
- [Yelp dataset 2017](https://www.yelp.com/dataset)<br>
This dataset contains a JSON of nearly 5 million entries. I splitted this JSON for performance reason to randomly 200000 train and 50000 test documents. I selected ratings with 1-2 stars as negative and 4-5 as positive. Ratings with 3 stars are not considered because of their neutrality. In addition comes that this selected subset contains only texts with more than 5 words. The language of the texts include english, german, spanish and a lot more. During the training I used 80% / 20% (train / validation). If you are interested you can also check a small demo of the [embeddings](https://github.com/cmasch/word-embeddings-from-scratch) created from the training data.

## Model
The implemented [model](https://github.com/cmasch/cnn-text-classification/blob/master/cnn_model.py) has multiple convolutional layers in parallel to obtain several features of one text. Through different kernel sizes of each convolution layer the window size varies and the text will be read with a n-gram approach. The default values are 3 convolution layers with kernel size of 3, 4 and 5.<br>

I also used pre-trained embedding [GloVe](https://nlp.stanford.edu/projects/glove/) with 300 dimensional vectors and 6B tokens to show that unsupervised learning of words can have a positive effect on neural nets.

## Results
For all runs I used a learning rate reduction if their's no improvement on validation loss by factor 0.1 after 4 epochs. The optimizer for all runs was Adadelta.<br>As already described I used 5 runs to get a final mean of loss / accuracy.

### Sentence polarity dataset v1.0
| Filter Sizes | Feature Maps | Embedding | Max Words / Sequence | Batch Size / Epochs | Training<br>(loss / acc) | Validation<br>(loss / acc) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [3,4,5] | [10,10,10] | GloVe 300 | 15000 / 20 | 100 / 80 | 0.3505 / 0.8720 | 0.4688 / 0.7974 |
| [3,4,5] | [10,10,10] | 300 | 15000 / 20 | 100 / 80 | 0.3560 / 0.8763 | 0.5243 / 0.7786 |

### IMDB
| Filter Sizes | Feature Maps | Embedding | Max Words / Sequence | Batch Size /<br>Epochs | Training<br>(loss / acc) | Validation<br>(loss / acc) | Test<br>(loss / acc) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [3,4,5] | [10,10,10] | GloVe 300 | 15000 / 200 | 100 / 80 | 0.2289 / 0.9213 | 0.2963 / 0.8888 | 0.2994 / 0.8896 |
| [3,4,5] | [10,10,10] | 300 | 15000 / 200 | 100 / 80 | 0.2166 / 0.9305 | 0.3043 / 0.8883 | 0.3322 / 0.8751 |

### Yelp 2017
| Filter Sizes | Feature Maps | Embedding | Max Words / Sequence | Batch Size /<br>Epochs | Training<br>(loss / acc) | Validation<br>(loss / acc) | Test<br>(loss / acc) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--: |
| [3,4,5] | [10,10,10] | GloVe 300 | 15000 / 200 | 200 / 40 | 0.1733 / 0.9407 | 0.1724 / 0.9418 | 0.1793 / 0.9393 |
| [3,4,5] | [10,10,10] | 300 | 15000 / 200 | 200 / 40 | 0.1251 / 0.9583 | 0.1647 / 0.9424 | 0.1753 / 0.9384 |

### Yelp 2017 - Multiclass classification
All previous evaluations are typical binary classification tasks. The Yelp dataset comes with reviews which can be classified into five classes (one to five stars). For the evaluations above I merged one and two star reviews together to the negative class. Reviews with four and five stars are labeled as positive reviews. Neutral reviews with three stars are not considered. In this evaluation I trained the model on all five classes.
The baseline we have to reach is 20% accuracy because all classes are balanced to the same amount of samples. In a first evaluation I reached 60% accuracy. This sounds a little bit low but you have to keep in mind that in the binary classification we have a baseline of 50% accuracy. That is more than twice as much! Furthermore there is a lot subjectivity in the reviews. Take a look on the confusion matrix:

<img src="./images/yelp_confusion.png">

If you look carefully you can see that it’s hard to distinguish in one class that has surrounding classes side by side. If you wrote a negative review, when does this have just two stars and not one or three?! Sometimes it’s clear for sure but sometimes not! Therefore I calculated another result with an approach of smooth transition (+-1) for each class. If we do so, we get an accuracy of 94.71% which is similar to binary classification but not very valid.

| Filter Sizes | Feature Maps | Embedding | Max Words / Sequence | Batch Size /<br>Epochs | Training<br>(loss / acc) | Validation<br>(loss / acc) | Test<br>(loss / acc) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--: |
| [3,4,5] | [10,10,10] | 300 | 15000 / 200 | 200 / 50 | 0.8613 / 0.6395 | 0.9036 / 0.6179 | 0.9356 / 0.6051 |

## Conclusion and improvements
Finally CNNs are a great approach for text classification. However a lot of data is needed for training a good model. It would be interesting to compare this results with a typical machine learning approach. I expect that using ML for all datasets except Yelp getting similar results. If you evaluate your own architecture (neural network), I recommend using IMDB or Yelp because of their amount of data.<br>

Using pre-trained embeddings like GloVe improved accuracy by about 1-2% especially for small datasets. In addition comes that pre-trained embeddings have a regularization effect on training. That make sense because GloVe is trained on data which is some different to Yelp and the other datasets. This means that during training the weights of the pre-trained embedding will be updated. You can see the regularization effect in the following image:

<img src="./images/yelp_comparison.png">

If you look on the results using GloVe you can see that training / validation / test are very close to each other.

I tried to modify just a few parameters for each dataset. Increasing batch size, filter size and sequence length need a lot memory and time to train. Therefore I tried to use small values. Maybe increasing this parameters can improve results like feature maps to 100 as written in the paper [1].

If you are interested in CNN and text classification try out the dataset from Yelp! Not only because of the best result in accuracy, it has a lot metadata. Maybe I will use this dataset to get insights for my next travel :)

I'm sure that you can get better results by tuning some parameters:
- Increase feature maps
- Add / remove filter sizes
- Use another embeddings (e.g. Google word2vec)
- Increase / decrease maximum words in vocabulary and sequence
- Modify the method `clean_text`

If you have any questions or hints for improvement contact me through an issue. Thanks!

## Requirements
* Python 3.6
* Keras 2.0.8
* TensorFlow 1.1
* Scikit 0.19.1

## Usage
Feel free to use the [model](https://github.com/cmasch/cnn-text-classification/blob/master/cnn_model.py) and your own dataset. As an example you can use this [evaluation notebook](https://github.com/cmasch/cnn-text-classification/blob/master/Evaluation.ipynb).

## References
[1] [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)<br>
[2] [Neural Document Embeddings for Intensive Care Patient Mortality Prediction](https://arxiv.org/abs/1612.00467)

## Author
Christopher Masch
