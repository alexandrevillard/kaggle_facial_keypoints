# Kaggle Facial Keypoints
In this project, I will use Google's Tensorflow library to solve the [Facial Keypoints](https://www.kaggle.com/c/facial-keypoints-detection) challenge which was hosted on Kaggle a few months back. This project has not for purpose to find the best possible model to obtain the best predictions in this particular problem (in fact, hyperparameters tuning will not be conducted in great depth, or at least not described here), but rather to give an end-to-end example of how Tensorflow can be used to quickly build a DNN model and train it on the cloud (see this [script](cloud.sh) to submit the training of the model on Google Cloud ML, especially for those like me who cannot run on a GPU).

It builds up on the work of [Alex Staravoitau](https://github.com/navoshta/kaggle-facial-keypoints-detection) which itself builds up on the work of [Daniel Nouri](http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/). We will obtain similar results with the following differences:
* Creating TFRecords files for the inputs
* Usage of Tensorflow's queues to read the data rather than placeholders
* Usage of MonitoredTrainingSession with hooks to perform evaluations and saving summaries
* Usage of an early stopper to detect overfitting
* Running the training on a Cloud ML instance

Results are reported in this [notebook](res_analysis.ipynb)
