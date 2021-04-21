import torch
from torchtext import data
import random
import torch.nn as nn
import torch.optim as optim
import spacy
nlp = spacy.load('en_core_web_sm')


def predict(model, smile_string, data_field):
    """
    Function to predict the outcome of varying topology RNNs.
    
    ARGS:
        - model: Pytorch model.
        - smile_string: A smile molecule representation in str format.
        - data_field: A data field object to get the corresponding index for each atom.
         
    OUTPUT:
        - Raw prediction. Should be passed to sigmoid/softmax function to get a probability
          or to exp() to get prediction for a Poisson neural network.
    """
    
    # In case dropout is used.
    model.eval()
    
    # Check on wich device we should do the computation.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Tokenize the smile string.
    tokenized = [tok.text for tok in nlp.tokenizer(smile_string)]
    
    # Retrieve the corresponding index for the embedding layer. 
    indexed = [data_field.vocab.stoi[t] for t in tokenized]
    
    # Get the length of the sequence so that we can use varying topology RNNs.
    length = [len(indexed)]
    
    # Change format and put into the desired device.
    tensor = torch.LongTensor(indexed).to(device)
    length_tensor = torch.LongTensor(length)
    
    # Input should be of size [Batch size, length].
    tensor = tensor.unsqueeze(1).T
    
    # Feed it to the network.
    prediction = model(tensor, length_tensor)
    
    return prediction.item()


def one_train(model, iterator, optimizer, criterion, accuracy_function=None):
    """
    Function to train varying topology RNN for one epoch.
    
    ARGS:
        - model: Pytorch model.
        - iterator: A BucketIterator with fields = [('text', <field1>), ('label', <field2>)].
        - optimizer: A Pytorch optimizer object such as SGD or Adam.
        - criterion: A Pytorch loss function.
    """

    # Needed to store losses and accuracies.
    epoch_loss = 0
    epoch_acc = 0
    
    # Needed to display progress of an epoch.
    counter = 0

    # Model in training stage
    model.train()

    for batch in iterator:
        
        # Display progress.
        counter += 1
        perc = round((counter/len(iterator))*100)
        print(f"Epoch progress : {perc}%                 \r", end="")

        # At each batch, we backpropagate the signal.
        optimizer.zero_grad()

        # Retrieve smile string and length.
        text, text_lengths = batch.text
        text_lengths = text_lengths.cpu()

        # Compute the prediction.
        predictions = model(text, text_lengths).squeeze()

        # Compute the loss.
        loss = criterion(predictions, batch.label)

        # Compute the gradients.
        loss.backward()

        # Backpropagate the signal.
        optimizer.step()

        # Store the loss.
        epoch_loss += loss.item()
        
        if accuracy_function is not None:
            
            # Compute the accuracy.
            acc = accuracy_function(predictions, batch.label)
            
            # Store the accuracy.
            epoch_acc += acc

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion, accuracy_function=None):
    """
    Function to evaluate varying topology RNN for one epoch.
    
    ARGS:
        - model: Pytorch model.
        - iterator: A BucketIterator with fields = [('text', <field1>), ('label', <field2>)].
        - criterion: A Pytorch loss function.
    """

    # Needed to store losses and accuracies.
    epoch_loss = 0
    epoch_acc = 0

    # Deactivating dropout layers.
    model.eval()

    # Deactivate gradient computation.
    with torch.no_grad():

        for batch in iterator:

            # Retrieve smile string and length.
            text, text_lengths = batch.text
            text_lengths = text_lengths.cpu()

            # Compute the prediction.
            predictions = model(text, text_lengths).squeeze()

            # Compute the loss.
            loss = criterion(predictions, batch.label)

            # Store the loss and accuracy.
            epoch_loss += loss.item()
            
            if accuracy_function is not None:

                # Compute the accuracy.
                acc = accuracy_function(predictions, batch.label)
                
                # Store the accuracy.
                epoch_acc += acc

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def train(model,
          train_iterator,
          valid_iterator,
          n_epoch,
          optimizer,
          criterion,
          accuracy_function=None,
          save=False,
          saving_path="_.bin"):
    """
    Function to train varying topology RNN for <n_epoch> epoch.
    
    ARGS:
        - model: Pytorch model.
        - train_iterator: A BucketIterator with fields = [('text', <field1>), ('label', <field2>)] for training data.
        - valid_iterator: A BucketIterator with fields = [('text', <field1>), ('label', <field2>)] for validation data.
        - n_epoch: Number of epoch to train.
        - save: If True to model is saved in bin format in <saving_path>
        - saving_path: Saving path for the model.
        - lr: Learning rate for the optimizer.
    """
    
    # Check on wich device we should do the computation.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Start by assuming the best validation loss is inf, so that it can be replaced at first epoch.
    best_valid_loss = float('inf')
    
    # Put model and criterion to the desired device.
    model = model.to(device)
    criterion = criterion.to(device)

    for epoch in range(n_epoch):

        train_loss, train_acc = one_train(model, train_iterator, optimizer, criterion, accuracy_function=accuracy_function)

        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, accuracy_function=accuracy_function)
        
        # Save if we beat the current best validation loss.
        if save:
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), saving_path)
        
        if accuracy_function is not None:
            
            # Quick summary.
            print(f"\tNo. epoch: {epoch+1}/{n_epoch}                  ")
            print(f'\t   Train Loss: {round(train_loss, 3)} | Train Acc: {round(train_acc, 3)}%')
            print(f'\t    Val. Loss: {round(valid_loss, 3)} |  Val. Acc: {round(valid_acc, 3)}%')
            print("\n")
            
        else:
            
            # Quick summary.
            print(f"\tNo. epoch: {epoch+1}/{n_epoch}                  ")
            print(f'\t   Train Loss: {round(train_loss, 3)}')
            print(f'\t    Val. Loss: {round(valid_loss, 3)}')
            print("\n")



class leaky_lstm(nn.Module):
    """
    LSTM model with Leaky relu MLP on top of it.
    """

    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 hidden_size,
                 n_classes,
                 dropout=0,
                 bidirectional=False):

        super().__init__()
        
        """
        ARGS:
            - num_embeddings: Number of embeddings, used to initialize the embedding layer.
            - embedding_dim: Size of token representation.
            - hidden_size: Size of hidden state representation.
            - n_classes: Number of classes. Used to quickly switch from regression to classification.
            - dropout: Dropout probability.
            - bidirectional: If True, the model is bidirectional.
        """

        # Needed to adapt the classifier if we have bidirectionality.
        self.bidirectional = bidirectional

        # Match word index to word vector.
        self.embedding = nn.Embedding(num_embeddings,
                                      embedding_dim)

        # LSTM cell.
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_size,
                            batch_first=True,
                            bidirectional=bidirectional)

        # Adding MLP on top of the non-linear transformations.
        bidirectional_factor = (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(hidden_size*bidirectional_factor, hidden_size*bidirectional_factor)
        self.linear2 = nn.Linear(hidden_size*bidirectional_factor, hidden_size*bidirectional_factor)
        self.linear3 = nn.Linear(hidden_size*bidirectional_factor, n_classes)

    def forward(self, word_indices, lengths):

        # Looking for vectors.
        embedded = self.embedding(word_indices)

        # Packed sequence to implement dynamic/varying topology LSTM.
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True)

        # Implicit recurrence.
        _, (hidden, cell_state) = self.lstm(packed_embedded)

        # Leaky relu MLP on concatenated hidden states.
        hidden = (torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1) if self.bidirectional else hidden)
        hidden = self.dropout(hidden)
        linear_combinations1 = nn.functional.leaky_relu(self.linear1(hidden))
        linear_combinations2 = nn.functional.leaky_relu(self.linear2(linear_combinations1))
        linear_combinations3 = self.linear3(linear_combinations2)

        return linear_combinations3