# Capsule-Network
 Capsule networks are used for understanding the spatial information, as they help in making sense of words even if they are not adjacent, which makes the approach not one of the words to word analysis but provides the classifier to understand the context of the entire network.
A capsule consists of a group of neurons along with the activation, which is the vector representation of instantiation parameters which helps in providing the probability for the existence of an entity in a piece of textual information.
Capsule network uses the dynamic routing algorithm which helps in preserving not only one, but all the useful features for a sentence or a part of the text.


## Requirements
* Anaconda
* Keras <br>
You can either run the code on python 3.6 or on the kaggle kernel GPU.

  
## Usage
Run the files in the following order:
1. **Libraries.py** contains all the libraries to be imported initially. 

2. **Dataset.py** consists of the importing the training and testing datasets. For the training dataset, rows containing "no answer" are removed. The dataset is imported in the kaggle kernel format. You can change it for running on python 3.6

3. **Preprocessing.py** has all the functions for preprocessing the dataset. <br>
The ‘question’ and ‘specification’ columns were merged together into a new column named ‘question_text’. For preprocessing the data, first, the short forms of words are changed to their original forms such as changing ‘cause’ to ‘because’ or ‘i’m’ to ‘I am’. After correcting such spellings removal of the punctuations like{ ',', '.', '"', ':', ')', '(',} is carried out. Next, the numbers such as{1,2,3} and special characters such as {@, #, $}are removed along with stop words like {is, are, will}. All the missing values in the dataset are then replaced by a random string.

4. **Embedding.py** contains word embeddings. <br>
For the embedding matrix, I used **glove embedding** and **fasttext embedding**. The glove pretrained vector can be found on <https://nlp.stanford.edu/projects/glove/> and the fasttext pretrained vector from <https://fasttext.cc/docs/en/english-vectors.html>.
 The final embedding matrix was calculated by taking the mean of the glove and fasttext embedding. 

5. **Capsule.py** consists of the capsule layer where the activation function **squash** was used to bring the output vector in the range between 0 and 1. The parameters are set as follows:
  Routings = 3
 * Num_capsule = 10
 * Dim_capsule = 16
 * initializer='glorot_uniform'
 
 6. **Train.py** has the get_model() function which consist of the bidirectional GRU(), capsule(), flatten(), and model.compile() and model.fit().
 The GRU length is set as 256 with the dropout as 0.25 and dropout rate as 0.28.
 The model.compile() has the following parameters:
  * loss='binary_crossentropy'
  * optimizer=optimizers.Adam(Lr=0.001)
  * metrics=['accuracy']
  * batch_size=412
  * epochs=30 <br>
 Here, Lr is the learning rate which is set to 0.001
 
 7. **Predict.py** contains the model.predict() and the softmax() function. <br>
 It calculates the number of hits at various indexes of the sorted list of lists of probabilities.
 
## Reference

* **Investigating Capsule Networks with Dynamic Routing for Text Classification** <br>
  Authors: Zhao, Wei and Ye, Jianbo and Yang, Min and Lei, Zeyang and Zhang, Suofei and Zhao, Zhou <br>
  Journal: arXiv preprint arXiv:1804.00538 <br>
  Year: 2018 <br>
  Link : <https://arxiv.org/abs/1804.00538>
  
* **Dynamic Routing Between Capsules** <br>
  Authors: Sara Sabour, Nicholas Frosst, Geoffrey E Hinton <br>
  Journal : arXiv:1710.09829 <br>
  Year: 2017 <br>
  Link: <https://arxiv.org/abs/1804.00538>

