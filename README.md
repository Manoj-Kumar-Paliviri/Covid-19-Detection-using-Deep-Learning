# Covid-19-Detection-using-Deep-Learning
COVID-19, typically known as Coronavirus, is an infectious disease caused by
a newly discovered coronavirus, which has emerged in the Republic of China for an
undetermined cause and has quickly affected the whole world. It is important to detect
positive cases early to prevent the further spread of the outbreak. To test a
In COVID-19 patients, a healthcare provider uses a long swab to take a nasal sample. The
diagnosis becomes even more critical when, there is a lack of reagents or testing
capacity, to track the virus and its severity, and the risk of a healthcare practitioner
getting affected when he comes in contact with COVID-19-positive patients. In this
scenario of the COVID-19 pandemic, there is a need for streaming diagnosis based on
a retrospective study of laboratory data in the form of chest X-rays using deep learning.

The dataset used in this project consists of X-ray images which are provided by
Kaggle.com and other Github resources to train and test the deep learning
model. This dataset consists of 3 categories viz. COVID, Normal, Viral Pneumonia in
the training and testing data. The total number of images taken is 15,000 which
is further divided into two parts training set and a testing set with a ratio of 4:1 i.e.;
The training set has 12,000 samples and the testing set has 3000 samples. In support
of this project, we have used libraries viz. TensorFlow, Keras, and its applications of
predefined model architectures like ResNet50 or VGG16, matplotlib, NumPy,
EarlyStopping, ModelCheckpoint, streamlit for the Image visualization, analytics, and
saving the optimized model respectively. The source code is written in a Python
environment. After training the model, to prove real-time efficiency, we
evaluated our model to check its performance by plotting the trained model’s accuracy
and loss concerning the validated model. The accuracy of the validated model was
found to be around 94 %. Further, the deployment is achieved through a local host.

Early diagnosis is essential both for early intervention of the patient and to prevent the
risk of transmission. For this purpose, chest X-ray images were used, obtained from
Covid-19 and non-Covid-19 patients. After the positive diagnosis of COVID-19
patients, we aim to track the progression of the disease which may help healthcare
professionals work on the correct dynamics and treatment of patients. It can also be
used in situations where the possibilities are insufficient (RT-PCR test, doctor,
radiologist). In future work, more successful deep-learning models can be created.

**Run DL_Streamlit.py to deploy it on the local host.**
