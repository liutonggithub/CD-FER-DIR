# FER

##Pre-requisites

 (1) Python 2.7
 
 (2) Scipy
 
 (3) Tensorflows (r0.12)
 
 ##Datasets
 
 (1) You may use any dataset with labels of expression and domain. In our experiments, we use JAFFE, Oulu-CASIA, RAF-DB and SFEW2.0. 
 
 (2) It is better to detect the face before you train the model. In this paper, we use a face detection algorithm (https://github.com/deepinsight/insightface)

 ##Training
 ```
 $ python maincdfer.py
 ```

 During training, two new folders named 'FER' and 'result', and one text named 'testname.txt' will be created. 

 'FER': including four sub-folders: 'checkpoint', 'test', 'samples', and 'summary'.

 (1) 'checkpoint' saves the model
 
 (2) 'test' saves the testing results at each epoch (generated fourier phase information reconstructed facial images based on the input faces).
 
 (3) 'samples' saves the reconstructed facial images at each epoch on the sample images. 
 
 (4) 'summary' saves the batch-wise losses and intermediate outputs. To visualize the summary.
 
 *You can use tensorboard to visualize the summary*
 
```
 $ cd PFER/summary
 $ tensorboard --logidr 
```

 *After training, you can check the folders 'samples' and 'test' to visualize the reconstruction and testing performance, respectively.*
 
 'result': including three kinds of '.txt' files: '*.txt', '*index.txt', and '*test.txt', where * means the number of epoch.
 
 (1) '*.txt' saves the training results at each epoch on the training data. 
 
 (2) '*index.txt' saves the classified labels for each test image.
 
 (3) '*test.txt' saves the number of the test images that classified right.
 

 'testname.txt': including the name of all the test images, which you can use to calculate the accuracy over each expression and domain. 

##Testing

```
$ python maincdfer.py --is_train False --testdir path/to/your/test/subset/
```

Then, it is supposed to print out the following message.

****
Building graph ...

Testing Mode

Loading pre-trained model ...
SUCCESS

Generate the images reconstructed with Fourier phase information.

Done! Results are saved as FER/test/test_as_xxx.png
***

##Files

(1) 'CD_Facial_expression_train.py' is a class that builds and initializes the model, and implements training and testing related stuff.

(2) 'ops.py' consists of functions required in 'CD_Facial_expression_train.py' to implement options of convolution, deconvolution, fully connection, max_pool, avg_pool, leaky Relu, and so on.

(3) 'maincdfer.py' demonstrates 'CD_Facial_expression_train.py'.
