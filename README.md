# ADHD-fMRI-Classification-With-ConvLSTM
 
## Motivation
- Diagnosing mental disorders is a considerably complex task for behavioral health professionals
- Many factors complicate the process:
 - People exhibit individual behaviors with the symptoms of their disorder(s)
 - No objective biological markers associated with mental disorders
 - Mental disorders often overlap with one to many others 
 - Similarity of symptoms among different diseases can lead to inaccurate diagnosis 
- Potential Solution lies with functional Magnetic Resonance Imaging (fMRI) technology

## functional Magnetic Resonance Imaging
- Measures brain activity by detecting changes associated with blood flow
 - Known as blood-oxygen-level dependent (BOLD) method
- This technique relies on the fact that blood flow and neuronal activation are coupled. 
- When an area of the brain is in use, blood flow to that region also increases 
- Does so to provide energy to the neurons, which do not have internal reserves of energy.
- fMRI machine captures blood flow and “lights up” brain areas in images
- Indicates that part of the brain is responsible for handling a certain activity

## Objective
- Construct a hybrid model that captures and analyzes both the spatial and temporal aspects of an fMRI dataset
- This model will consist of a Convolutional Neural Network and Recurrent Neural Network
- Convolutional Neural Network:
 - Retrieves spatial features of the data
 - Extracts the details of active areas in the brain
- Recurrent Neural Network 
 - Retrieves the temporal features of the data
 - Model the flow of the blood that is associated to certain disorders (or activities)
- Goal: Construct a 3D Convolutional Neural Network + LSTM-based RNN to diagnose the ADHD disorder with the provided fMRI data sample

## Currently In Progress
- The model architecture is fully implemented and undergoing a 5-fold cross-validation analysis 
- Implemented using Keras with a Tensorflow Backend
- Dataset can be found here: http://preprocessed-connectomes-project.org/adhd200/index.html



