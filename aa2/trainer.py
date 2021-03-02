# This class is suppose to handle the training, saving, loading and testing of your models.
# You will have to finish train(), test() and load_model() while save_model() is already made (this means you have to create load_model() so that it fits with how models are saved).

import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score

class Trainer:


    #def __init__(self, dump_folder="/tmp/aa2_models/"):
    def __init__(self, dump_folder):
        # create a directory "aa2_models" for storing the models:
        self.dump_folder = dump_folder
        os.makedirs(dump_folder, exist_ok=True)


    def save_model(self, epoch, model, optimizer, loss, scores, hyperparamaters, model_name):
        # epoch = epoch
        # model =  a train pytroch model
        # optimizer = a pytorch Optimizer
        # loss = loss (detach it from GPU)
        # scores = dict where keys are names of metrics and values the value for the metric
        # hyperparamaters = dict of hyperparamaters
        # model_name = name of the model you have trained, make this name unique for each hyperparamater.  I suggest you name them:
        # model_1, model_2 etc 
        #  
        #
        # More info about saving and loading here:
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training

        save_dict = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'hyperparamaters': hyperparamaters,
                        'loss': loss,
                        'scores': scores,
                        'model_name': model_name
                        }

        torch.save(save_dict, os.path.join(self.dump_folder, model_name + ".pt"))


    def load_model(self, model_path): #, model, optimizer):
        # Finish this function so that it loads a model and return the appropriate variables
        
        #model = TheModelClass(*args, **kwargs)
        #optimizer = TheOptimizerClass(*args, **kwargs)

        checkpoint = torch.load(model_path)
        #model.load_state_dict(checkpoint['model_state_dict'])
        #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #epoch = checkpoint['epoch']
        #loss = checkpoint['loss']

        return checkpoint #, epoch, loss


    def train(self, train_X, train_y, val_X, val_y, model_class, hyperparameters):
        # Finish this function so that it set up model then trains and saves it.       
        
        lr = hyperparameters["learning_rate"]
        n_layers = hyperparameters["number_layers"]
        optimizer_choice = hyperparameters["optimizer"]
        batch_size = hyperparameters["batch_size"]
        epochs = hyperparameters["epochs"]
        model_name = hyperparameters["model_name"]

        train_X = train_X.unsqueeze(2)
        train_y = train_y.unsqueeze(2)
        val_X = val_X.unsqueeze(2)
        val_y = val_y.unsqueeze(2)
        
        input_size = train_X.shape[2]  # no of features: 1 (POS tags)
        sample_size = train_X.shape[1]
        output_size = 6 # number of ner labels
        hidden_size = hyperparameters["hidden_size"]
        
        # declare device (i.e. GPU) 
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        
        #create mini batches out of train_X and train_y
        mini_batches = Batcher(train_X, train_y, device, batch_size=batch_size, max_iter=epochs)

        # initiate model
        model = model_class(input_size, hidden_size, output_size, n_layers)
        print("model: ", model)
        
        # move the model to the GPU
        model.to(device)
        
        # all models use same optimizer
        if optimizer_choice == "adam":
            optimizer = optim.Adam(model.parameters(), lr=lr)
        elif optimizer_choice == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=lr) 
        
        # loss function
        criterion = nn.NLLLoss()
        
        epoch = 0
        for split in mini_batches:
            # go into training mode
            model.train()
            tot_loss = 0
            for sentence_x, label_y in split: # sentence_x = [32,165,1], label_y = [32,165,1]
                # Since the backward() function accumulates gradients, and you don’t want to mix up gradients between 
                # minibatches, you have to zero them out at the start of a new minibatch using zero_grad():
                optimizer.zero_grad()
                pred = model(sentence_x.float(), device) # pred = [32,165,6]
                #pred = pred.permute(0, 2, 1)  # pred after permute = [32,6,165]
                #label_y = label_y.squeeze(2)  # label_y after squeeze: [32,165] 
                loss = criterion(pred.float(), label_y).to(device)
                tot_loss += loss
                loss.backward()
                optimizer.step()
            print("Total loss in epoch {} is {}.".format(epoch, tot_loss))
            epoch += 1
    
        model.eval()
        y_label = []
        y_pred = []
        val_batches = Batcher(val_X, val_y, device, batch_size=batch_size, max_iter=1)
        for split in val_batches:
            for sentence, label in split:
                with torch.no_grad():
                    pred = model(sentence.float(), device)
                    for i in range(pred.shape[0]):
                        pred_s = pred[i]
                        label_s = label[i]
                        for j in range(len(pred_s)):
                            pred_t = int(torch.argmax(pred_s[j]))
                            label_t = int(label_s[j])
                            y_pred.append(pred_t)
                            y_label.append(label_t)
    
        scores = {}
        accuracy = accuracy_score(y_label, y_pred, normalize=True)
        scores['accuracy'] = accuracy
        recall = recall_score(y_label, y_pred, average='weighted')
        scores['recall'] = recall
        precision = precision_score(y_label, y_pred, average='weighted')
        scores['precision'] = precision
        f = f1_score(y_pred, y_label, average='weighted')
        scores['f1_score'] = f
        scores['loss'] = int(tot_loss)
            
        self.save_model(epoch, model, optimizer, tot_loss, scores, hyperparameters, model_name)
            
        print('model:', model_name, 'accuracy:', accuracy, 'precision:', precision, 'recall:', recall, 'f1_score:', f)
        
        pass

    def test(self, test_X, test_y, model_class, best_model_path):
        # Finish this function so that it loads a model, test is and print results.
        lr = hyperparameters["learning_rate"]
        n_layers = hyperparameters["number_layers"]
        optimizer_choice = hyperparameters["optimizer"]
        batch_size = hyperparameters["batch_size"]
        epochs = hyperparameters["epochs"]
        model_name = hyperparameters["model_name"]
        
        print("trainx_shape:", train_X.shape)
        inputsize = train_X.shape[2]  # = 7, number of features per word
        samplesize = train_X.shape[1]
        outputsize = 6 # number of ner labels
        hiddensize = hyperparameters["hidden_size"]
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        train_batches = Batcher(train_X, train_y, self.device, batch_size=batch_size, max_iter=epochs)
        model = model_class(inputsize, hiddensize, outputsize, n_layers)
        model = model.to(self.device)
        
        
        if optimizer_choice == "adam":
            optimizer = optim.Adam(model.parameters(), lr=lr)
        elif optimizer_choice == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=lr) 
            
        criterion = nn.NLLLoss()
        
        epoch = 0
        for split in train_batches:
            model.train()
            tot_loss = 0
            for sentence, label in split:
                optimizer.zero_grad()
                pred = model(sentence.float(), self.device)
                pred = pred.permute(0, 2, 1)        
                loss = criterion(pred.float(), label).to(self.device)
                tot_loss += loss
                loss.backward()
                optimizer.step()
            print("Total loss in epoch {} is {}.".format(epoch, tot_loss))
            epoch += 1
           
            
            model.eval()
            y_label = []
            y_pred = []
            test_batches = Batcher(val_X, val_y, self.device, batch_size=batch_size, max_iter=1)
            for split in test_batches:
                for sentence, label in split:
                    with torch.no_grad():
                        pred = model(sentence.float(), self.device)
                        for i in range(pred.shape[0]):
                            pred_s = pred[i]
                            label_s = label[i]
                            for j in range(len(pred_s)):
                                pred_t = int(torch.argmax(pred_s[j]))
                                label_t = int(label_s[j])
                                y_pred.append(pred_t)
                                y_label.append(label_t)
                     
    
            scores = {}
            accuracy = accuracy_score(y_label, y_pred, normalize=True)
            scores['accuracy'] = accuracy
            recall = recall_score(y_label, y_pred, average='weighted')
            scores['recall'] = recall
            precision = precision_score(y_label, y_pred, average='weighted')
            scores['precision'] = precision
            f = f1_score(y_pred, y_label, average='weighted')
            scores['f1_score'] = f
            scores['loss'] = int(tot_loss)
        

            print('model:', model_name, 'accuracy:', accuracy, 'precision:', precision, 'recall:', recall, 'f1_score:', f)

        
            self.save_model(epoch, model, optimizer, tot_loss, scores, hyperparameters, model_name)

        pass

class Batcher:
    def __init__(self, X, y, device, batch_size=50, max_iter=None):
        self.X = X
        self.y = y
        self.device = device
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.curr_iter = 0
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.curr_iter == self.max_iter:
            raise StopIteration
        permutation = torch.randperm(self.X.size()[0], device=self.device)
        permX = self.X[permutation]
        permy = self.y[permutation]
        splitX = torch.split(permX, self.batch_size)
        splity = torch.split(permy, self.batch_size)
        
        self.curr_iter += 1
        return zip(splitX, splity)