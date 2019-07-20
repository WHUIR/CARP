# CARP
The implementation of “A Capsule Network for Recommendation and Explaining What You Like and Dislike”, Chenliang Li, Cong Quan, Li Peng, Yunwei Qi, Yuming Deng, Libing Wu

## Requirement
Tensorflow 1.2/1.4

Python 2.7

Numpy

Scipy

## Data Preparation
To run CARL, 5 files are required:

### Training Rating records:
file_name=TrainInteraction.out

each training sample is a sequence as:

UserId\tItemId\tRating\tDate

Example: 0\t3\t5.0\t1393545600

### Testing (Validate) Rating records:

file_name=TestInteraction.out 

The format is the same as the training data format.

### Word2Id diction:

file_name=WordDict.out

Each line follows the format as:

Word\tWord_Id

Example: love\t0

### User Review Document:

file_name=UserReviews.out

each line is the format as:

UserId\tWord1 Word2 Word3 …

Example:0\tI love to eat hamburger …

### Item Review Document:

file_name=ItemReviews.out

The format is the same as the user review doc format.

## Note that:
All files need to be located in the same directory. We also provide the data preprocessing code (Java implementation) for the Amazon datasets. The code can directly output the required data files once you download the original data file from http://jmcauley.ucsd.edu/data/amazon/index.html  (K-core data) and pass it to the preprocessing code. The preprocessing of other datasets follows the same steps.

Carp_runner.py is the implementation of  CARP model; Note that by substituting the function named caps_layer_2 to caps_layer_1, you can get the implementation of CARP-RA which use the vanilla dynamic routing mechanism.

## Configurations
word_latent_dim: the dimension size of word embedding;

latent_dim: the latent dimension of the sentiment representation learned from CARP, denotes as k in paper;

max_doc_length: the maximum doc length;

num_filters: the number of filters of convolution operation;

window_size: the length of the sliding window of CNN;

learn_rate: learning rate;

lambda_1: the weight to control the impact of the mutual exclusion in sentiment classification task;

drop_out: the keep probability of the drop out strategy;

batch_size: batch size;

epochs: number of training epoch;

itr_1: the number of the iteration of Dynamic Routing in Capsule

game: use to control the trade-off in the two training task, denotes as λ
in paper

number_aspect: number of viewpoint/aspect, denotes as M in paper

rating_thrhld: the threshold to partition the rating into positive and negative (higher than thrhld = positive)
