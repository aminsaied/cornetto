# The Cornetto Classification Library
Welcome to cornetto! The aim of this library is to build a mathematics classifier. More specifically, we build a variety of machine learning models, including:  

- neural networks (fully connected/RNNs)
- SVMs
- naive Bayes models

designed to read a mathematics text, such as an abstract of a paper, and to predict its subject. We choose to use the standard [Mathematics Subject Classification](https://en.wikipedia.org/wiki/Mathematics_Subject_Classification) (henceforth, MSC) to enumerate our subjects.

Please note: this library is still under construction.

## Our Data
We built a database from the mathematics arxiv (https://arxiv.org/archive/math) consisting of:

- Title
- Authors
- Abstract
- Date
- MSC code (this is an optional field in arXiv!)  

We found that _only about half_ of the papers on the arXiv have actually been labelled with their MSC code. We use this half to train our models and test our models. Furthermore, we have built a website where you can use our models to classify your own texts and abstracts. Check it out [here](http://classifythatabstract.herokuapp.com/) (please note that this is still under construction).

Our code is written in Python 3. We have written a series of modules and scripts to facilitate:

- scraping data from the web
- processing data
- handling data
- training models
- using trained models

Libraries we use heavily: numpy, pandas, tensorflow, scikit-learn.
