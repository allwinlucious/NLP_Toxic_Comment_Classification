
<h1><center>Toxic Comment Classification</center></h1>


_Repo metadata_


[![GitHub release](https://img.shields.io/github/release/mewbot97/NLP_Toxic_Comment_Classification?include_prereleases=&sort=semver&color=blue)](https://github.com/mewbot97/NLP_Toxic_Comment_Classification/releases/)
[![License](https://img.shields.io/badge/License-MIT-blue)](#license)
[![License]](https://mewbot97-nlp-toxic-comment-classification-webapp-rsxwz6.streamlitapp.com)


# Introduction
<p style='text-align: justify;'> With increased accessibility to the internet, communication has crossed all the barriers. Nowadays, people can share thoughts whenever, wherever with the entire world through social media. These comments not only reach people who favor them but also to people who are against them, and thus, give rise to open debates. Since the explicit interaction is with a computer or smartphone and not with humans, people tend to forget proper behaviour while commenting and type abusive toxic comments. Many times, people misbehave with this technology by creating fake accounts to spread hate and negativity.
As it is said, diamond cuts a diamond, we want to use technology to reduce the abuse of technology. This project is about creating a Toxic Comment Filter Bot using Data Science and Machine Learning methods, namely Natural Language Processing. NLP enables computers to process human language in the form of text or voice data and to ‘understand’ its full meaning, complete with the speaker or writer’s intent and sentiment.[1] The purpose of this bot is to identify threats and abusive comments, which are then categorized in six different categories based on the toxicity level. We want to enable this bot, in future, to be used as an extension with any social media to filter or hide out the toxic comments.</p>

# Method
<p style='text-align: justify;'>Therefore, classifying inappropriate comments can be a solution for a user friendly social platform. Here, around 160000 Wikipedia comments which have been labeled by humans into 6 different classes, like toxic, severe toxic, obscene, threat, insult, and identity hate are available from Kaggle competition.  Moreover, the correlation for each class is shown in picture below with pearson method. It is interesting to see that, “toxic” and “severe toxic” has less correlation compare to “toxic” with “obscene” or to “insult”. Higher correlation between “toxic” with “obscene” could be explained, that each sentence has both of labels. Furthermore, “threat” has less correlation with other classes compare to others as shown in image below.</p>
<br />
<p align="center">
    <img src="images\matrix.PNG">
    <br>
    <em>Image 1. Correlation of each label</em>
</p>
<br />
<p style='text-align: justify;'>Moreover, the dataset has mostly “toxic” labels with around 15000 entries and the lowest is “threat” labels, with only around 1000 entries. This imbalance has to be taken into consideration before the classification step, because it influences how well the classification model performs. Many options, like upsampling/downsampling or data augmentation could be used to solve this imbalance.</p>
<br />
<p align="center">
    <img src="images\class_dis.png">
    <br>
    <em>Image 2. Class distribution</em>
</p>
<br />

# Data preprocessing
<p style='text-align: justify;'>The steps for data preprocessing and cleaning will be divided into five small steps. First, lower case the sentence from <em>“ She eats Äpple and orange!”</em> to <em>“ she eats äpple and orange!”</em>. Next, unnecessary symbol or characters, like !,@,” are removed, so that at the end the sentence will be “she eats äpple and orange”. Here, the symbol database from non-ASCII text and regular expression (RegEx) is used to make sure all such symbols are removed. Accented words that can be found for example in German, like äpfel  is also changed to apfel. In this English dataset, the anomaly is usually just a misspelling and it is not common to use accented words. The difference between this algorithm and other NLP data cleaning alogirthms is that we keep the “negative” words like <em>not, hadn’t or shouldn’t</em>, because these toxic comments have high correlation with these words and  “not” is  also the most common word in dataset. Stopwords, like “and” or “or” will be also removed because these words do not provide further information. In this algorithm , NLTK is chosen as a library for stopwords and other text processing. Therefore, we don’t need to specify every word, especially in English. Unnecessary white space in the sentence is also deleted. Lastly, lemmatization is performed for every word in dataset, which means the word is returned to its base form, like eats to eat, and talking to talk. At the end of the entire preprocessing, the example sentence will become <em>“she eat apple orange”</em>.</p>
<br />

# Classification Model
<p style='text-align: justify;'>In this project, long short term memory (LSTM) is used to classify multiple label dataset. LSTM  is a type of recurrent neural network and better than traditional recurrent neural network in terms of memory, because it has multiple hidden layer. Moreover, LSTM is also better than the traditional machine learning models, like random forest, especially in NLP because with LSTM, we can give a sentence as an input for prediction rather than just one word. Therefore, LSTM from keras library is chosen for this work with sigmoid for the last layer with binary cross entropy to calculate the loss. To use the model, the sentence needs to be tokenized with a maximum of 200 words for each sentence. Tokenization is the process of exchanging sensitive data for nonsensitive data called "tokens" [2].</p>
<br />

# Conclusion
<p style='text-align: justify;'>The LSTM model is trained with 25 epochs to see whether the model is overfitting or not. To train the model, it took around 1 hour with Intel i5 CPU and the highest accuracy with 99% accuracy is achieved with 2 epochs. After 2 epochs, the data was overfitting. Moreover, the output of the model will be a percentage from 0 to 100% for each label, because every comment could have multiple labels. However, the model’s self developed spell check functions take extremely long time to execute. Implementing an optimal and faster method for spell check can improve the accuracy and efficiency of the model.
Moreover, it can be seen in class distribution, the amount of entries per class is still imbalanced, i.e., presence of very few ‘interesting’ events (toxic/hate comments in our case). Most of the ML algorithms do not work well with imbalanced datasets. Some of the techniques (or combinations) that can be tried out to handle imbalanced dataset are using the correct Evaluation Matrix, resampling the training set, resample with different ratios, cluster the abundant class and various Data Augmentation techniques.
As compared to Computer Vision, where transformations are done on the go using data generators, data augmentation should be done carefully in  NLP due to the grammatical structure of the text. Back translation, EDA (Easy Data Augmentation), NLP Albumentation, NLP Aug are few methods that can be used [3]. Future possibilities for the Toxic Comment Filter Bot includes enabling it as an extension for various social media platforms such as twitter, facebook etc. Moreover, in future, it  can be configured to work in real time to avoid people from even posting toxic comments.</p>
<br />

Github Repo:<br />
https://github.com/mewbot97/NLP_Toxic_Comment_Classification/blob/main/Toxic%20Comment%20Classification.ipynb<br />


References:<br />
[1] https://www.ibm.com/cloud/learn/natural-language-processing<br />
[2] https://www.tokenex.com/resource-center/what-is-tokenization<br />
[3] https://neptune.ai/blog/data-augmentation-nlp<br />



