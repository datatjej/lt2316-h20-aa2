## LT2316 H20 Assignment A2 : Ner Classification

Name: *fill in your real name here* (if you don't want to use your real name with your current GitHub account, you will have to make another GitHub account)

## Notes on Part 1.

*fill in notes and documentation for part 1 as mentioned in the assignment description*

## Notes on Part 2.

*fill in notes and documentation for part 2 as mentioned in the assignment description*

## Notes on Part 3.

*fill in notes and documentation for part 3 as mentioned in the assignment description*

## Notes on Part 4.

*fill in notes and documentation for part 4 as mentioned in the assignment description*


## Notes on Part Bonus.

*fill in notes and documentation for the bonus as mentioned in the assignment description, if you choose to do the bonus*

------------------------------------------------------

NER Classification

In this assignment we will use the preprocessed data from Assignment 1 to create, train and evaluate a NER classifier.

This assignment is structured in the same way as Assignment 1 in such a way that parts of the code will be given, not only to enforce a structure, but to help and give you ideas of to structure machine learning project. It will be your job to fill in the missing parts, some of which will not be given explicitly. 

The task is to make run.ipynb run which outline a workflow for how to arrive at a good set of hyperparameters for a model, also called hyperparameter tuning.

NOTE, the task is to make run.ipynb run. How you make it run with the pieces you are given is up to you. You will have to figure out what you need to add and make decisions about how to add it. There is no one true way of doing this. You will not get penalized for a decision you make if it does not 1) stop run.ipynb from running or 2) fails to cover the missing part (i.e. trainer.train() needs to actually train your model).  For example, in Part 2, you have to evaluate a model, this means choosing a metric. The run.ipynb will run with any metric (accuracy, f1, AUC-ROC etc), so what metric to use it up to you! Furthermore, how and where you calculate the metric; aggregate outputs over steps and then calculate or calculate the metric for each step and then aggregate the score before the next epoch/on the last step, is up to you as both will work with run.ipynb.

You must start by forking the GitHub repository (https://github.com/LT2316-H20/lt2316-h20-aa2Länkar till en externa sida.) and then checking run.ipynb. Most work done in this assignment will be to make the code in this notebook run. This will give you a good idea of what is needed to be done. Note that you are not allowed to change anything in run.ipynb if its not explicitly said so.

Next step is to replace the aa1 folder in assignment A2 with your assignment A1 aa1 folder so that we can use the pipeline you created. The easiest way to do this is simply to copy and paste.

NOTE: This assignment is actually a part of a series of connected assignments. In Assignment A3, you will use and expand on the code from this assignment.

You can discuss this assignment and ask questions on the Assignment A2 discussion thread: Assignment A2 discussion.

Submitted assignments must run on the mltgpu server, even if you developed them on your own computer.

The official due date for this assignment is 9th of October 2020 at 23:59. You will submit a link to your forked GitHub repository. There are 26 points available on this assignment plus 4 bonus points.
Part 1: Create a model (10 Points)

The first part of this assignment will be to create a deep learning classifier in PyTorch. You are free to create any type of network you like as long as it contains at least 1  of the following Pytroch Networks/layers:

    RNNLänkar till en externa sida.
    LSTMLänkar till en externa sida.
    GRULänkar till en externa sida.

The classifier is suppose to take a sequence of words and return a sequence of labels, one label per token.

This model should be placed hereLänkar till en externa sida. and imported in run.ipynb.

Examples of PyTorch models can be found in Demo 1.1 and 1.2 (and matching video material). Here is also a link to a PyTorch guide: https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-custom-nn-modulesLänkar till en externa sida.

Motivate the choice of network and describe the architecture.
Part 2: Finishing the Trainer class (10 points)

The second part of the assignment is to finish TrainerLänkar till en externa sida.. This class is suppose to handle the training, saving, loading and testing of your models. If you look in the class you will find four functions train(), test(), load_model(), save_model(). You will have to finish train(), test() and load_model() while save_model() is already made (this means you have to create load_model() so that it fits with how models are saved).

train() should set up the model with the hyperparameters and then continue to train and validate the model. When a model is finished it should save the model with the appropriate information as specified in save_model().

test() should load a model from a file, test the model and return scores.

We suggest that you create a helper function which you can reuse for training, validation and testing which takes care of everything that happens within an epoch.

Look through at Demo 1.1 and 1.2 for examples how to set up training and evaluation. You can also look here: https://pytorch.org/tutorials/beginner/pytorch_with_examples.htmlLänkar till en externa sida.
Part 3: Hyperparameter tuning (5 points)

Now that you have a model defined and the trainer set up you need to create a set of hyperparameters. Some hyperparameters you can focus on:

    learning rate
    hidden size
    number layers
    optimizer
    batch size

Test at least 5 sets of hyperparameters.

After you are done use parallel_coordinates()Länkar till en externa sida. to create a parallel coordinates plot to visualize the performance of your hyperparameter sets.

more on Parallel Coordinate plots:

https://datavizcatalogue.com/methods/parallel_coordinates.htmlLänkar till en externa sida.

https://ai.facebook.com/blog/hiplot-high-dimensional-interactive-plots-made-easy/ Länkar till en externa sida.

 

parallel_coordinates()Länkar till en externa sida. will default use loss as the last Y-axis,  feel free to change this to any other metric that you think would fit better, such as e.g. F1.

Next step is to define 3 new sets of hyperparameters based on the parallel coordinate plot. Train models for each of these hyperparameters and run the plot again.

Document which hyperparameters you are testing and why, if you have chosen them for any particular reason. Did the parallel coordinate plot help at all with the tuning of hyperparameters? Did you find anything interesting?
Part 4: Test the best model (1 points)

Now when you have done a few experiment with different hyperparameters chose the best model and pass the file to test() to test the model on the test set.

Document which hyperparameters were best, their test scores, and how much different they were from the validation scores.
Bonus Part: Oh no wait, what have we done!! (4 points)

As you might have noticed, the dataset includes named entities that stretch across multiple tokens, e.g., “trimethyl lead”. At the moment, we would recognize such spans by a sequence of labels. Let's look at the following sentence:

    DDI-MedLine.d93.s4. "Thirty male rats of the Fischer-344 strain were divided into three equal groups and were given injections of trimethyl lead (TML) (8.0 or 17.0 mg/kg/ml SC) or the saline vehicle. "

Depending on your tokenization and cleaning we will get different results here, but lets assume we are tokenizing by spaces and removing parenthesis. Let's label the words around “trimethyl lead”:

1A. ".. were given injections of trimethyl lead (TML) ... " 0 0 0 0 1 1 1

Now, how would you extract the 2 separate NEs “trimethyl lead” and “TML”? Currently, our labels contain no information that help us figure out where one NE starts and another ends. Lets look at some other problematic examples:

    DDI-MedLine.d81.s1. "Decreased core temperature in female rats was investigated as one possible index of the disulfiram-ethanol reaction (DER). "

Here “disulfiram-ethanol” are two separate NEs:

    DDI-MedLine.d128.s1: "The 16,16-dimethylprostaglandin E2 (dmPGE2)-induced diarrhea was analyzed in cecectomized rats prepared by resecting the cecum and its vasculature without disturbing the ileocecal junction. "

Here “16,16-dimethylprostaglandin E2” is one NE.

So, how can we solve these cases? How can we set up the task in such a way that we get information about where a NE starts and ends? In sequence tagging, one usually uses label a encoding scheme called IOB (or BIO), where I is for Inside , O is for Outside and B is for Beginning. If we apply this encoding to our labels, we can enrich our models with information about the beginnings and ends of NEs. For example, for LABEL_X, we would now have two labels, B-LABEL_X and I-LABEL_X, while O would work for any item not belonging to any label. If we revisit the example above with our new label encodings we get:

1A. ".. were given injections of trimethyl lead (TML) ... " O O O O B-1 I-1 B-1

And now we can separate the NEs. BIO is not the only encoding one can use; more complex ones can be more useful for certain tasks. See these links for some more alternative encodings:

https://lingpipe-blog.com/2009/10/14/coding-chunkers-as-taggers-io-bio-bmewo-and-bmewo/Länkar till en externa sida.

https://www.aclweb.org/anthology/D16-1008.pdfLänkar till en externa sida.

https://www.aclweb.org/anthology/W15-5603.pdfLänkar till en externa sida.

 

 

Try to find a sentence in the test which contains two NEs which your classifier is currently unable to separate, write the predictions out, choose an encoding scheme, and correct the labels by hand. Simply mirror what we have done above! Motivate your choice of encoding scheme and try to find cases which the encoding scheme would not be able to deal with.
