# Kaggle Facial Keypoints
In this project, I will use Google's Tensorflow library to solve the [Facial Keypoints](https://www.kaggle.com/c/facial-keypoints-detection) challenge which was hosted on Kaggle a few months back. I did this project mainly to gain some hands-on experience with Tensorflow. 

It builds up on the work of [Alex Staravoitau](https://github.com/navoshta/kaggle-facial-keypoints-detection) which itself builds up on the work of [Daniel Nouri](http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/). We will obtain similar results with the following differences:
* Creating TFRecords files for the inputs
* Usage of Tensorflow's queues to read the data rather than placeholders
* Usage of MonitoredTrainingSession with hooks to perform evaluations and saving summaries
* Usage of an early stopper to detect overfitting
* Running the training on a Cloud ML instance

Results are reported in this [notebook](res_analysis.ipynb)
