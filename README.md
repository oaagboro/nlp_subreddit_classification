## NLP CLASSIFICATION USING TWO SUBREDDITS

This project involves using NLP and Classification Models to predict which subreddit a post came from based on its title and text.

## Problem Statement
Given a piece of text within the title and the original post from r/harrypotter and r/thewalkingdeas can we predict which subreddit the post cam from with an accuracy that is greater than 85%.

## Data Source
The data used for this project was collected using the PushShift API: [Pushshift's](https://github.com/pushshift/api).
5000 posts were collected in total, 2500 posts from each subreddit incliding the title ans the selftext. These were then concatenated into one dataframe and used for the analysis.

## Executive Summary
In order to solve the task as described by the Problem Statement, PushShift's API was utilized to collect 5000 subreddits in total, 2500 each from r/harrypotter and r/thewalkingdead. Most of the other columns were dropped leaving the 'selftext' and 'title' which were then concatenated into a single text column to use for analysis.

After this feature engineering, Exploratory data analysis was then carried out and 2 interesting charts were developed in the process targeting the subreddits: r/harrypotter and r/thewalkingdead.

A baseline model was then determined and it scored 0 on the test set.

The next step involved modelling. First a pipeline was set up with a gridsearch to find the best Logistic Regression with a Countvectorizer transformation. The following hyperparameters were optimized:

<ul>
    <li>CountVectorizer: max_features, stop_word, ngram_range, analyzer(PorterStemmer, Lemmatizer, default)
    <li>LogicticRegression: C, penalty
</ul>
        


Using a CV=5, this pipeline fit 720 models and ran for about 40 mins. In the end, LogisticRegression using TfidfVectorizer gave very similar results with the CountVectorizer using same model. And just like the CountVectorizer it did overfit slightly. However the only similarities in the parameters were the ngram_range and the max_features which were (1,1) and 500 respectively.
    
Modelling was also carried out using the Multinomial Naive Bayes. The data was transformed using both the CountVectorizer and the TfidfVectorizer on separate occasions. In the end, the TfidfVectorized model did better than the CountVectorized model in terms of Accuracy and Precision.
    
## Model Performance
Below is a table summarizing the performance of the models that came out of the pipelines. Generally the Multinomial Naive Bayes performed better than the Logistic Regression on both with both Vectorizers. All four models were able to achieve greater than the expected accuracy of 85%, but were slightly overfit. 

|Vectorizer|Estimator|Test Accuracy|Train Accuracy| Train Precision|
|---------|---------|---------|---------|---------|
|Countvectorizer|Logistic Regression|0.942|0.975|0.926|
|Countvectorizer| Multinomial Naive Bayes|0.957|0.969|0.945|
|Tfidfvectorizer|Logistic Regression|0.946|0.952|0.949|
Tffidfvectorizer|Multinomial Naive Bayes|0.962|0.97|0.955|

## Conclusions
For the given problem statement, we were able to achieve over 85% in terms of accuracy. However, the models were slightly overfit in all cases with the Train data accuracy greater than the test data accuracy. The Tfidfvectorized models performed better than the Countvectorized models on both cases. 

## Next Steps
In order to solve the overfitting problem, more data would need to be obtained. Then the procedure would involve trying to perform the modelling process first without Stemming and Lemmatization, then use Stemming and Lemmatization and compare the results of both cases and find out the effect of Stemming and Lemmatization on the modelling proces.