# <p style="text-align: center;">Sentiment Analysis and Visualization of Real and Fake Reviews.</p>

This project aims to detect fake reviews and perform sentiment analysis using various machine learning techniques. By analyzing text data from reviews, we can identify fraudulent reviews and determine the sentiment expressed.

## Project Overview
### Dependencies:
This project relies on several Python libraries for data processing, machine learning, and visualization:

* pandas, numpy, seaborn, matplotlib for data manipulation and visualization.
* nltk (Natural Language Toolkit) for natural language processing tasks.
* scikit-learn for machine learning algorithms and evaluation metrics.
* wordcloud for visualizing word frequencies.
  #### Setup
   * **Install Dependencies:** Ensure all required libraries are installed.
   * **Download NLTK Data:** Obtain the necessary NLTK datasets, such as stopwords.
### Data Processing:
* **Text Tokenization:** Break down the reviews into individual tokens (words) using NLTK's tokenizer.
* **Stopword Removal:** Filter out common stopwords that do not contribute significant meaning.
* **Stemming:** Reduce words to their root form using the PorterStemmer to standardize text.
### Feature Extraction:
**Count Vectorizer:** Convert the textual data into a numerical matrix of token counts, facilitating model training.
### Model Building:
We utilize several machine learning classifiers to build and evaluate models for fake review detection and sentiment analysis:

* **RandomForestClassifier:** A versatile classifier using multiple decision trees.
* **LogisticRegression:** A linear model for binary classification tasks.
* **DecisionTreeClassifier:** A simple yet powerful tree-based model.
* **Support Vector Machine (SVM):** A robust classifier for high-dimensional data.
* **Multinomial Naive Bayes (MultinomialNB):** A probabilistic classifier for text data.
* **XGBoost (XGBClassifier):** An efficient and scalable implementation of gradient boosting.
### Evaluation Metrics:
To assess model performance, we use several evaluation metrics:

* **Accuracy:** The proportion of correctly classified instances.
* **Precision:** The ratio of true positives to the sum of true and false positives.
* **Recall:** The ratio of true positives to the sum of true positives and false negatives.
* **F1 Score:** The harmonic mean of precision and recall.
* **Confusion Matrix:** A matrix displaying the true versus predicted classifications.
### Visualization:
* **Word Cloud:** Generate a word cloud to visualize the most frequent words in the reviews.
* **Confusion Matrix Display:** Display the confusion matrix to analyze model performance.
### Results:
The results section will present the performance metrics for the trained models, showcasing their effectiveness in detecting fake reviews and analyzing sentiment. Visualizations such as word clouds and confusion matrices will provide additional insights into the model's behavior and the data's characteristics.We also seperated the fake and real data based on the sentiment.


