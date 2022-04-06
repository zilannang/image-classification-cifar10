## image-classification

In this work, Convulutional Neural Network (CNN) which is called Deep Learning algorithm applies image classification through CIFAR-10 in order to compare our models results. In the experimental part, the results of the proposed models in terms of accuracy and loss values in the test data according to the data sets used are shown graphically when the models complete the test on the basis of the optimum epoch numbers. At the end of the experimental study, it was recorded that the accuracy values were increasing and the loss values were decreasing gradually up with models are well trained and tested.

 ### Dataset
  
The CIFAR-10 dataset contains 6000 images per class, consisting of 10 classes of 60000 32x32 color images in total. There are 50000 training images and 10000 test images. The dataset is divided into five training (data_batch_1, data_batch_2, ..., data_batch_5) and a test (test_batch) set, each containing 10000 images. The test set contains exactly 1000 images selected randomly from each class. The remaining images of the training sets are generated in random order, but some training datasets may contain more images from one class than the other. Between them, the training sets contain exactly 5000 images from each class.
  
 ###Performance Metrics
   
The models created in this study should be evaluated and two important performance metrics, accuracy and loss, should be calculated.
First, the models are compiled to obtain the loss value. Loss is a quantitative measure of the difference or deviation between the predicted output and the expected actual output. It gives us the measure of the errors made by the network in predicting the output. In other words, the loss value is a measure of how well the model performed during the testing phase. A low loss value means the model is good. In this study, categorical cross entropy was used as loss function.

### Tools and Libraries

The processing of our dataset, the creation, training and testing of our models are coded using the Python programming language and the necessary environment and libraries. Here, the virtual environment in the visual studio code was used as the environment and the libraries were Pandas, Numpy, Mathplotlib, Sklearn, Scikit-Learn, TensorFlow and Keras. The first thing to do to train our models is to separate the datasets. In our study, in training (x_train, y_train) or (x_test, y_test), a certain part of the training sets is reserved for validation and testing (x_test, y_test) or (test_images, test_labels).

### Confusion Matrix (Hata Matrisi)
    
In order to evaluate the performance of classification models used in machine learning, the error matrix, which compares the predictions of the target attribute and the actual values, is often used.

### Model and Results

| Model | Accuracy value |
|--|--|
| LeNet | 0.7291 |
| AlexNet | 0.7103 |
| VGG | 0.7291 |

