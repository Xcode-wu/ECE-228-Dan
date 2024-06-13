# Video to Commands for Robotic Manipulator

## Introduction
In this project, we present an improved approach for translating video input into
executable commands for humanoid robots in a grammar-free setting. This project
addresses the limitation of traditional learning from demonstration (LD) meth-
ods, which is costly and requires extensive manual programming. Our improved
framework first uses state-of-the-art convolutional neural network (CNN) models
as deep visual feature extractors. The two LSTM models then served as sequential
encoders and decoders of the extracted visual feature. We also tested the model
with a transformer replacing the existing LSTM module. We pass the result into a
Softmax function, which generates captions as commands. Our experiments on the
Breakfast Action Dataset demonstrate the effectiveness of our improved structure
and models, as well as evaluating the CIDEr metric. The results of the experiment
show our improved model outperforms some recent models by a small margin.

## Steps to Run the Code
Run Code in Google Colab with instructions - 
https://colab.research.google.com/drive/1O7HVac3hPZ2mT4wiam6DSdLZB4OocXmz?usp=sharing


## References

This code was inspired from Authors code available here - https://github.com/nqanh/video2command and https://github.com/cjiang2/video2command. 
