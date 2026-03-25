# twitter_sentiment_classifier
Indonesian tweet sentiment classification project with EDA, text preprocessing, TF-IDF + Random Forest, and LSTM baseline.

# Indonesian Tweet Sentiment Classification

This repository contains an Indonesian tweet sentiment classification project covering exploratory data analysis (EDA), text preprocessing, baseline machine learning, and deep learning experiments on tweets posted during the 2019 Indonesian presidential election period.

The task is to classify tweets into three sentiment classes:

- `negatif`
- `netral`
- `positif`

## Project Overview

The project was built to compare two main approaches for sentiment classification on Indonesian 2019 tweets:

1. **TF-IDF + Random Forest**
2. **LSTM-based sequence model**

The workflow includes:

- data loading and quality checking
- exploratory data analysis
- text cleaning and preprocessing
- train/validation/test split with stratification
- baseline modeling
- hyperparameter tuning
- evaluation using accuracy, macro F1, classification report, and confusion matrix

## Dataset

The dataset contains **1,815 Indonesian tweets** with two main columns:

- `tweet`
- `sentimen`

### Label distribution

- `positif`: 612
- `netral`: 607
- `negatif`: 596

The dataset is relatively balanced, so accuracy is still meaningful, but **macro F1** is used as the main metric to keep evaluation fair across classes.

## Exploratory Data Analysis

Several EDA steps were performed before modeling:

- checking missing values and duplicate tweets
- inspecting class balance
- analyzing tweet length using character and word counts
- extracting top unigram and bigram frequencies
- generating a word cloud

### Key EDA findings

- No missing values
- No duplicate tweets
- Tweets vary quite widely in length
- The dataset is dominated by **Indonesian political-economic topics**
- Twitter-specific noise such as `pic twitter`, `twitter com`, and platform tokens appeared frequently
- Informal/slang words such as `yg`, `gk`, and `ga` were common
- The positive class tended to have slightly longer tweets on average
- Extreme outliers existed, so sequence length for the neural model was determined using **p95** instead of max length

## Text Preprocessing

The preprocessing pipeline was designed based on EDA findings.

### Main steps

- lowercase conversion
- URL removal
- mention removal
- hashtag normalization (`#word` -> `word`)
- non-alphanumeric character removal
- whitespace normalization
- slang normalization
- stopword removal
- preserving negation words important for sentiment
- removing Twitter/platform noise tokens

### Important normalization examples

- `yg` -> `yang`
- `dgn` -> `dengan`
- `tdk` -> `tidak`
- `ga`, `gak`, `gk`, `nggak` -> `tidak`

### Negation words intentionally preserved

Because negation is important in sentiment analysis, the following words were kept instead of removed:

- `tidak`
- `tak`
- `bukan`
- `jangan`
- `belum`
- `tanpa`

## Data Split

The dataset was split with **stratified sampling** to preserve label proportions across subsets:

- **Train**: 1452 samples
- **Validation**: 181 samples
- **Test**: 182 samples

A fixed random seed was used to make experiments reproducible.

## Models

### 1. TF-IDF + Random Forest

This model uses:

- TF-IDF features
- unigram and bigram representation for the baseline
- Random Forest classifier with class balancing

A tuning stage was later applied using `RandomizedSearchCV` and `StratifiedKFold` to improve generalization and reduce overfitting. The tuning process explored several combinations of:

- n-gram range
- minimum document frequency (`min_df`)
- maximum document frequency (`max_df`)
- maximum TF-IDF features
- number of trees (`n_estimators`)
- maximum tree depth
- minimum samples split
- minimum samples leaf
- feature selection strategy in Random Forest
- class weighting


### 2. LSTM

The deep learning approach uses:

- Keras `Tokenizer`
- padded integer sequences
- maximum sequence length determined from **95th percentile**
- embedding layer
- spatial dropout
- LSTM layer
- dense classification head

A tuning stage was also performed by testing several combinations of:

- embedding dimension
- LSTM units
- dense units
- dropout
- learning rate
- batch size

## Results

| Model | Test Accuracy | Test Macro F1 |
|------|--------------:|--------------:|
| Baseline TF-IDF + Random Forest | 0.5330 | 0.5244 |
| Tuned TF-IDF + Random Forest | 0.5055 | 0.4978 |
| Baseline LSTM | 0.4670 | 0.4140 |
| Tuned LSTM | 0.5385 | 0.5297 |

### Best tuned TF-IDF + Random Forest configuration

- `tfidf__ngram_range = (1, 1)`
- `tfidf__min_df = 3`
- `tfidf__max_features = 20000`
- `tfidf__max_df = 0.85`
- `rf__n_estimators = 300`
- `rf__min_samples_split = 5`
- `rf__min_samples_leaf = 1`
- `rf__max_features = log2`
- `rf__max_depth = 60`
- `rf__class_weight = balanced`

### Best tuned LSTM configuration

- `emb_dim = 128`
- `lstm_units = 128`
- `dense_units = 32`
- `dropout = 0.3`
- `lr = 0.0005`
- `batch_size = 32`

**Observation:**  
Tuning significantly improved LSTM performance compared to the baseline, including better recall for the positive class.

## Final Model

The **final selected model is Tuned LSTM**, because it achieved the best **test performance** among the final compared models:

- **Tuned LSTM** -> Test Accuracy: **0.5385**, Macro F1: **0.5297**
- **Tuned TF-IDF + Random Forest** -> Test Accuracy: **0.5055**, Macro F1: **0.4978**

Tthe tuned LSTM generalized better on the held-out test set, so it was chosen as the final model.

## Main Takeaways

- The dataset is fairly balanced, so macro F1 is a suitable main metric.
- EDA was important for identifying Twitter-specific noise and slang normalization needs.
- Random Forest with TF-IDF was highly prone to overfitting.
- LSTM required tuning, but after tuning it became the best final model.
- The **positive** class remained the hardest class to classify correctly.

## Future Improvements

Potential future work includes:

- trying models more suitable for sparse text features such as Logistic Regression or Linear SVM
- exploring stronger recurrent architectures such as BiLSTM
- adding attention mechanisms
- using pretrained embeddings such as Word2Vec
- comparing results with Indonesian transformer models such as IndoBERT
- conducting more robust evaluation with multiple splits or cross-validation
- performing deeper error analysis on the positive class
- using stemming and lematiation library for Indonesian like Sastrawi
- using wordninja library for more depth hashtag handling

## Tools and Libraries

Main libraries used in this project include:

- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `nltk`
- `tensorflow`
- `wordcloud`

## Notes

This repository focuses on building a complete sentiment classification pipeline starting from data analysis and preprocessing up to baseline comparison and final model selection. For a more detailed explanation of the workflow, exploratory data analysis, preprocessing steps, model experiments, and evaluation results, please refer to our Jupyter Notebook (`.ipynb`) files.

## Contributor (NLP Group C)
- Surya Dharma Putra
- Agil Setiawan
- Krisna Fery Rahmantya
- Khaerani Arista Dewi​
- Irfan Gani Alim​
