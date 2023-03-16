# SC1015 Project


## Overview
This repository contains the project files for the SC1015 mini-project.<br>
With the recent increases in fraud and scams around the world, and specifically job scams, we will use **Natural Language Processing (NLP) techniques to classify job listings as fraudulent or not.**
Listed here are the ipynb files used, which should be viewed in numerical order:<br>
1. [EDA](https://github.com/edward62740/sc1015-project/blob/master/EDA.ipynb)
2. [Data Cleaning, NLP](https://github.com/edward62740/sc1015-project/blob/master/Data%20Cleaning%20and%20Lemmatization.ipynb)
3. [Baseline Model - Random Forest](https://github.com/edward62740/sc1015-project/blob/master/Random%20Forest.ipynb)
4. [Support Vector Machine (SVM) model](https://github.com/edward62740/sc1015-project/blob/master/Support%20Vector%20Machine.ipynb)
5. [Recurrent Neural Network (RNN) model](https://github.com/edward62740/sc1015-project/blob/master/Recurrent%20Neural%20Network.ipynb)

## Dataset
The dataset used can be found [here](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction). It provides approx. 18k job descriptions, labelled as either fraudulent or not.

## Methodology
### Data Cleaning and Lemmatization
The dataset contains a mix of numbers (stored as strings), boolean values and strings. All the data in each row was concatenated into a single string, with the boolean values being encoded in the form yes/no_{column name}. The main column of interest is the [job]"description".
After this, the stop words were removed with NLTK, and the text was lemmatized with spaCy.
Next, there was some post-processing to further clean the data, such as normalizing case, removing hyperlinks, symbols etc.
### Random Forest Model (Baseline)
The random forest model is used to demonstrate that this particular task of fake job classification requires a more complex model, as random forest yields poor results even with max depth of 32.
### SVM Model
The cleaned text was vectorized with TF-IDF, class balanced with ADASYN, and used to create a SVC. It was found that for this particular dataset, the linear kernel is most performant, with C param of 0.1. 
### RNN Model
The cleaned text was class balanced with ADASYN, encoded with one-hot encoding, and fed into a RNN with GloVe used as an embedding layer. Specifically, the RNN utilizes bi-directional LSTM to help with memory in the time domain. The float [0,1] output of the RNN is then rounded to a boolean.
The results of the classification are as below.

## Results
Below is the table denoting various performance statistics for the respective models on the **test** set (for the positive class, i.e. fraudulent)<br>
| Model                    | Accuracy | F1   | Recall (TPR) | Specificity (TNR) | Precision |
|--------------------------|----------|------|--------------|-------------------|-----------|
| LinearSVC                | 0.988    | 0.88 | 0.827        | 0.997             | 0.93      |
| Bi-directional LSTM      | 0.985    | 0.82 | 0.73         | 0.997             | 0.93      |
| Random Forest (Baseline) | 0.978    | 0.72 | 0.57         | 0.99              |           |

## Conclusion
In conclusion, we found that it is possible to classify fake job listings with relatively high accuracy (>80%), despite the highly imbalanced classes. More importantly, all the models are able to hit almost 100% accuracy with identifying real jobs, which is important as creating a model that falsely categorizes job listings as fraudulent will not be in the public interest.

Based on the successes of the ML models, we can determine that there exists a strong correlation between a real job listing and a certain pattern/word use in the job description.

## Contributors



## License
