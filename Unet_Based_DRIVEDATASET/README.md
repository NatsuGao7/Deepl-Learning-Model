# Unet for the Retinal Photography based on DRIVE DATASET

***
## Environment Configuration
* Python --version 3.9.15<br>
* Pytorch --version 1.13+cu117<br>
* You'd better use Ubuntu and the author use the Ubuntu version is 22.04.1 LTS<br>
* Specfic environmrnt configuration please reference the requirements.txt<br>
  Please use the pip install -r requirements.txtx to isntall
  
***
## The Structure of the PROJECT

* Please reference the structure.txt<br>
* src: Building the UNET Model
* train_utils: The module of train and evaluation and calculate the Dice_Coeffication_Loss
* train.py: Using the single GPU to train the model
* predict: Using the best weights to predict the the retinal photography
* results20230116-232259.txt and the test_result.png: Using bilinear to replace the Transposed Convolution to predict
* results20230124-103802.txt and the tests_result.png: Using Transposed Convolution to predict
  
***
## Download the DRIVE DATASET
* Introduction address: <https://drive.grand-challenge.org/>
* Download address: <https://drive.grand-challenge.org/Download/>

***
## Train Method
* Ensure that the data set is prepared in advance
* If you wanna use the SINGLE GPU please use the train.py to train
  
***
## Notes
* When using the training script, be careful to set the --data-path to the root directory where you store you DRIVE folder
* When using the prediction script, set the weights_path to your own generated weight path.
* When using the validation file, be careful to ensure that your validation set or test set must contain targets for each class, and that only the --num-classes, --data-path and --weights are modified when using them, leaving the rest of the code as unchanged as possible
  
***
## Contact  Information
* If you have any questions, please contact us and my email is<Natsugao0218@gmail.com><br>

