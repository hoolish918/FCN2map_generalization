# FCN2map_generalization
a method to construct and train a FCN to do building generalization in cartography

The aim of the repositor is to train a FCN to realize transforming automaticly building maps under the scale of 1:2000 into 1:10000. 
The construct of the net refers to the article: BUILDING GENERALIZATION USING DEEP LEARNING( https://www.int-arch-photogramm-remote-sens-spatial-inf-sci.net/XLII-4/565/2018/ ).

Note that this repositor is based on pytorch.

fcn8 is the model of Fully Convolution Net of 8-skipped connectted;
fcn16 is 16-skipped connectted;
fcn32 is 32-skipped connectted.
It is noted that fcn8 is able to get better results.

train is used to train the net. run train, and you can get the trained model.

Data is saved in 128乘128, 128乘128/data is the maps under the scale of 1:2000; 128乘128/label is the maps under the scale of 1:10000.
