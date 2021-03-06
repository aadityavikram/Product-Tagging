# Product Tagging

<h2> This is the project for tagging products based on their category for e commerce websites. Labels are predicted for the product image provided. </h2>

<h2> Using the Product Tagging.ipynb notebook :- </h2>
<p> Go to https://colab.research.google.com/ and create a new account. </p>
<p> Open this notebook in Google Colab. </p>
<p> From Edit -> Notebook Settings select Hardware Accelerator. Select GPU or TPU for fast processing. </p>
<p> Create a folder with the name 'ProductTagging' in the root of Google Drive. </p>
<p> Create a folder with name 'dataset' and another with name 'output' in ProductTagging folder. </p>
<p> Now copy the contents from the link -> https://drive.google.com/open?id=1cDrs72XW7j6eBtV-oLgsn7G4r9ZrUtV1 into the 'dataset' folder. </p>
<p> Now run the notebook. </p>

<h2> Final accuracy :- </h2>
<p> <b> Before Augmentation -> 82.46% </b> </p>
<p> <b> After Augmentation -> 94.26% </b> </p>

<h2> Setup :- </h2>
<p> Python -- > Python 3.6.5 </p>
<p> OS --> Windows 10 (OS build-->17763.253) (Version-->1809) </p>
<p> GPU --> Nvidia Geforce GTX 1060 (6gb) </p>
<p> CPU --> Intel Core i7-8750 @ 2.20GHz </p>
<p> RAM --> 16gb </p>

<h2> Step 1 -> Data Cleaning and Preparation :- </h2>
<h3>The image dataset can be cleaned and prepared from productsDataset.json file by running clean_prep.py</h3>
<p>python clean_prep.py</p>
<p>Creates dataset folder in project folder with two subfolders -> train and test.</p>
<p>Images used for training will be downloaded in dataset/train folder.</p>
<p>Images used for testing can be put in dataset/test folder.</p>
<p>Checks for the products from the json file with the correct tag format, eg -> OUTERWEAR_JACKETS.</p>
<p>Checks for the products which contain atleast one image.</p>
<p>Downloads those images in their respective categories in dataset/train.</p>
<p>Discards other images.</p>

<h2> Step 2 -> Data Augmentation :- </h2>
<h3>The image dataset can be augmented by running augmentation.py</h3>
<p>python augmentation.py</p>
<p>Increases the size of dataset if sufficient amount of data is not present for training.</p>
<p>scikit-image library in python can be used to augment the dataset.</p>
<p>Various operations are performed on images for augmentation.</p>
<p>Some of the operations are :-</p>
<table>
  <tr>
    <th>S.No.</th>
    <th>Operation to be performed</th>
    <th>Command</th>
  </tr>
  <tr>
    <td>1</td>
    <td>Rotating the images.</td>
    <td>skimage.transform.rotate(image_array, random_degree)</td>
  </tr>
  <tr>
    <td>2</td>
    <td>Adding Noise to the images.</td>
    <td>skimage.util.random_noise(image_array)</td>
  </tr>
  <tr>
    <td>3</td>
    <td>Flipping the images.</td>
    <td>image_array[:, ::-1]</td>
  </tr>
  <tr>
    <td>4</td>
    <td>Rescaling the images.</td>
    <td>skimage.transform.rescale(image_array, 1.0/4.0, anti_aliasing=False)</td>
  </tr>
</table>

<h2> Step 3 -> Extracting features from images in training dataset :- </h2>
<h3>The features can be extracted from the image dataset by running extract_features.py</h3>
<p>python extract_features.py</p>
<p>Creates output folder in project folder.</p>
<p>Labels, Features, Models and Weights of images in our dataset are saved in output folder.</p>

<h2>Step 4 -> Training on the image dataset :-</h2>
<h3>The model can be trained by running train.py</h3>
<p>python train.py</p>
<p>Pre-trained VGG16 model is used.</p>
<p>All the layers of VGG16 other than top fully connected layer are left intact.</p>
<p>The fully connected layer is retrained on our dataset.</p>
<p>Dataset is split into 70% training data and 30% testing data.</p>
<p>Logistic Regression is used to retrain the last layer.</p>
<p>Could use a neural network to retrain the last layer in the future.</p>
<p>Classifier is saved in output folder.</p>

<img src="extract_features.png">

<h2>Step 5 -> Testing on new data :-</h2>
<h3>The model can be tested by running test.py</h3>
<p>python test.py</p>
<p>Put images to be tested in dataset/test folder.</p>
<p>Saved classifier is loaded.</p>
<p>Image is loaded and resized to 224x224 as input to VGG16 is of this size.</p>
<p>Features are extracted.</p>
<p>Prediction is made on test images put in dataset/test folder.</p>
<p>Predictions are printed on screen and saved in predicted_labels.txt (created automatically) in the project folder as well.</p>

<h2> Links referenced :- </h2>
<p> https://keras.io/applications/#vgg16 </p>
<p> https://stackoverflow.com/questions/988228/convert-a-string-representation-of-a-dictionary-to-a-dictionary </p>
<p> https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html </p>
<p> https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html </p>
<p> https://towardsdatascience.com/logistic-regression-using-python-sklearn-numpy-mnist-handwriting-recognition-matplotlib-a6b31e2b166a </p>
<p> https://www.learnopencv.com/keras-tutorial-transfer-learning-using-pre-trained-models/ </p>
<p> https://www.analyticsvidhya.com/blog/2018/07/top-10-pretrained-models-get-started-deep-learning-part-1-computer-vision/ </p>
<p> https://pillow.readthedocs.io/en/3.1.x/reference/Image.html </p>
<p> https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#0 </p>
