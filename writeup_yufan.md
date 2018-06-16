# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./report_image/data_visualization.png "Visualization"
[image2]: ./report_image/before_preprocess.png "Before Preprocess"
[image3]: ./report_image/after_preprocess.png "After Preprocess"
[image4]: ./web_images/1.png "Traffic Sign 1"
[image5]: ./web_images/12.png "Traffic Sign 2"
[image6]: ./web_images/15.png "Traffic Sign 3"
[image7]: ./web_images/25.png "Traffic Sign 4"
[image8]: ./web_images/31.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/AfterDreaming/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is: 34799
* The size of the validation set is: 4410
* The size of test set is: 12630
* The shape of a traffic sign image is: (32,32,3)
* The number of unique classes/labels in the data set is: 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed. In general, each label has a fair amount of data examples but the different labeled examples  are not evenly distributed, for example, lable 2 and 3 have more than 1750 examples while label 41, 42 only have 210 examples. Because there aren't many examples to train(only 34799 training examples for 43 labels), the final model may have less accurary when detecting traffic sign of label with few examples.

![data distribution][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I normalized the image, because of the characteristics of the activation functions. Most of activation function(for example, tanh, and ReLU) have big difference between the left and right side of the origin point. Normalization the images helps the activation function work better.

I did not apply grayscale to the images. The reason is that the grayscale is a feature which can be derived from the image itself and the deep learning model can find it by itself if it is helpful. Although I didn't think grayscale will help, I still tried this method. The final result was not improve which confirm my theory.

Here is an example of a traffic sign image before and after the image normalization.

Before:
![Before preprocess][image2]
![After preprocess][image3]



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The model is a modified version of LeNet.
My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6				    |
| Convolution 5x5	    | 1x1 stride,  outputs 10x10x16	     			|
| RELU		            |            									|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16				    |
| Fully connected		| 400.        									|
| RELU		            |            									|
| Fully connected		| 600.        									|
| RELU		            |            									|
| Fully connected		| 300.        									|
| RELU		            |            									|
| Fully connected		| 84.        									|
| Softmax				|            									|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer with batch size 128, 50 epoch, and .001 learning rate

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 100%
* validation set accuracy of 94.2%
* test set accuracy of 93.2%


If a well known architecture was chosen:
* What architecture was chosen?
** LeNet

* Why did you believe it would be relevant to the traffic sign application?
** LeNet is a well known architechture for image recognition and its structure is relative simple comparing with some hundred layers model. It is easy to train and effective. I think it is a good starting points for image related AI questions.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
** LeNet can get 89% accuracy across training, validation, and test set without any modification. The result is very impressive considering the limited data we have. Because the trainging set accuracy was not good enough, first, I added additional a convolution layer and pooling layer. However, the accuracy was not improved. The reason is because our images have small size(32*32) and two convolutional layers is enough for extracting image features. Therefore, I add an additional fully connected layer to find more relationship between image features. After I got 100% training accuracy, I added dropout in the model, however, the validation accuracy was not improved. Then I decide to remove the dropout since it is not very helpful in this question.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![31][image8]![1][image4] ![12][image5] ![15][image6] 
![25][image7] 

The first one have many details but it is unclease because of the image quality.
The second image might be difficult to classify because there is noisy background.
The third image might be difficult to classify because the background color is simiar to the most part of the sign.
The forth image should be easy to classify.
The fifth one contains lots of details but the image is quite blur.


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (30km/h)  | Speed limit (30km/h)  						| 
| Priority road     	| Priority road									|
| No vehicles		    | No vehicles     								|
| Road work				| Road work										|
| Wild animals crossing	| Right-of-way at the next intersection			|

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 93% accuracy. The reson is that test set has much more example and provides better accuracy estimate.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final ∂model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a Priority road (probability of 1), and the image does not match the prediction (it should be Wild animals crossing). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1        				| Priority road  								| 
| 0     				| Traffic signals 								|
| 0						| Wild animals crossing							|
| 0	      				| Speed limit (50km/h)							|
| 0				  	    | Stop    										|


For the second image, the model is relatively sure that this is a Speed limit (30km/h) (probability of 1), and the image does contain this traffic sign. The top five soft max probabilities were
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| Speed limit (30km/h)  						| 
| 0     				| Speed limit (20km/h) 							|
| 0						| Speed limit (70km/h)							|
| 0	      				| Speed limit (50km/h)			 				|
| 0			    		| Speed limit (120km/h)    						|


For the third image, the model is relatively sure that this is a Priority road (probability of 1), and the image does contain this traffic sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1        				| Priority road 								| 
| 0     				| End of no passing								|
| 0						| Stop											|
| 0	      				| No entry							 			|
| 0			   			| Keep right     								|


For the forth image, the model is relatively sure that this is a No vehicles (30km/h) (probability of .999), and the image does contain this traffic sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .999         			| No vehicles  									| 
| .0     				| Speed limit (80km/h)							|
| .0					| Speed limit (120km/h)							|
| .0	      			| Speed limit (70km/h)			 				|
| .0				    | Speed limit (30km/h)    						|

For the fifth image, the model is relatively sure that this is a Road work (probability of .998), and the image does contain this traffic sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .998         			| Road work  									| 
| .0009     			| Pedestrians									|
| .0002					| Speed limit (100km/h)							|
| .0	      			| Children crossing					 			|
| .0				    | Traffic signals     							|


















