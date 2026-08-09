# Infrared-Spectrometer-Neural-Net
Run the files in the Single-Label folder for classification of one group only. Multilabel classifies carbonyls, hydroxyls, amines, nitriles, alcohols, carboxylic acids, alkenes, and C-O bonds. Multilabel with UI has a UI element to upload images and predict the presence of functional groups in the image. The default model is a CNN called SpectrumCNN, however a dense network (SpectrumNet) and a larger dense network (SpectrumNetBigger) are also available. The CNN reaches around 93% accuracy, running for up to 100 epochs (there is early stopping to prevent overfitting), and unique thresholds for classifying each functional group. This model was trained on 2097 spectra from the Chemotion dataset and 40,000 simulated spectra for pretraining.

To run the model, run Main.py.

## UI Usage
1. Load the image file using the load image button
2. To calibrate the axis follow the instructions (click 1 = at 4000cm-1, click 2 = at 400cm-1, click 3 = at 100% transmittance, click = at 0% transmittance)
3. Press 'Digitise and Predict' to get predictions for each functional group within the image and their probabilities.

## Description - Based on MultiLabel UI
- Data_Prep.py - finds the .dx file for the spectra used in training and then builds datasets for pretraining and training.  
- Dataset.py - wraps the dataset for use with pytorch's DataLoader  
- Models.py - contains the dense SpectrumNet and SpectrumNetBigger networks, alongside the SpectrumCNN and focal loss models  
- Train.py - trains the model, evaluates accuracy on validation data, and then finds the best threshold for each label before testing on the test data  
- Digitise.py - takes an image, calibrates its axis before extracting the curve and returning the digitised wave  
- Predict.py - predicts the functional groups in the input using the model  
- Main.py - optional loading prediction assets from prior training, functions to train and pretrain the model, and enter jcamp or image files to predict  
- Spectrum_UI.py - uses tkinter for user interface from uploading the spectra, calibrating the axes, extracting the wave, and predicting the functional groups*  

- pretrained_weights.pt - pretrained weights generated after running pretrain_on_ibm in Main.py once  
- trained_weights.pt - trained weights of model generated after running Main.py once where TRAIN_MODEL = True  
- thresholds.json - individual thresholds for each label generated after running Main.py once where TRAIN_MODEL = True  

Pretraining Spectra: https://zenodo.org/records/16417648  
Training spectra: https://radar4chem.radar-service.eu/radar/en/dataset/OGoEQGlsZGElrgst  


## Problems
- False positives: images can sometimes give false positives
- Poor accuracy for alkenes and C-O
- Pretraining uses simulated data, training uses real experimental data, so spectra will differ

- Focal loss, positive weighting, pre-training and augmenting data were all unsuccessful in raising accuracy and I have hit a roadblock



*UI was created by Codex, but all backend code is my own
