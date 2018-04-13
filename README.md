# Cornetto
The purpose of this library is to build a mathematics classifier. You'll find scripts that:

- scrape and preprocess data from the [arXiv](https://arxiv.org/archive/math)
- build and train RNN models using TensorFlow
- build and train Unigram models
- compute statistical anaysis of text (e.g., TFIDF)

using our Cornetto library (see `modules`).

### Example - training our RNN
First you need to create the training data. Use the class `RNNTrainingData` from the `data_handlers.py` module.

~~~~
rnn_training_data = RNNTrainingData.build([...])
~~~~

See `data_handlers` for more details on the `build` method. This gets passed directly to our `RNNModel` class from `rnn_model.py`.

~~~~
model = RNNModel(model_params)
model.fit(rnn_training_data)
~~~~

You can save and load models with `model.save([...])` and `model.load([...])`.
