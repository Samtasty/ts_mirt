import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import torch
from torch.utils.data import DataLoader, TensorDataset
import copy

class DKT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Initialize the DKT model.
        
        Args:
            input_dim (int): Number of input features (questions * responses).
            hidden_dim (int): Number of hidden units in the RNN.
            output_dim (int): Number of output features (questions).
        """
        super(DKT, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Forward pass through the DKT model.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length, input_dim).
        
        Returns:
            Tensor: Output predictions of shape (batch_size, sequence_length, output_dim).
        """
        rnn_out, _ = self.rnn(x)  # rnn_out: (batch_size, sequence_length, hidden_dim)
        logits = self.fc(rnn_out)  # logits: (batch_size, sequence_length, output_dim)
        predictions = self.sigmoid(logits)  # predictions: (batch_size, sequence_length, output_dim)
        return predictions,rnn_out


def preprocess_learning_trace(learning_trace, num_items):
    """
    Convert learning trace into sequences for the DKT model.
    
    Args:
        learning_trace (list of tuples): [(item_id, correctness), ...]
        num_items (int): Total number of items.
    
    Returns:
        inputs (torch.Tensor): Input sequences (batch_size, seq_len, input_dim).
        targets (torch.Tensor): Target sequences (batch_size, seq_len, num_items).
    """
    seq_len = len(learning_trace)
    inputs = np.zeros((seq_len, num_items * 2))
    targets = np.zeros((seq_len, num_items))
    
    for t, (item_id, correctness) in enumerate(learning_trace):
        # One-hot encoding of item and correctness
        inputs[t, item_id + (correctness * num_items)] = 1
        # Target is the one-hot of the next item's correctness
        if t + 1 < seq_len:
            next_item_id, next_correctness = learning_trace[t + 1]
            targets[t, next_item_id] = next_correctness
    
    return torch.tensor(inputs, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)




def training_model(model,dic_students,corpus,learning_rate,epochs,batch_size):
    """
    Train the DKT model.
    
    Args:
        model (DKT): DKT model to train.
        dic_students (dict): Dictionary of students.
        dic_items (dict): Dictionary of items.
        learning_traces (list of list of tuples): Learning traces for training.
        learning_rate (float): Learning rate for optimization.
        epochs (int): Number of training epochs.
        batch_size (int): Number of sequences per batch.
    """
    # Initialize optimizer

    # Create a TensorDataset and DataLoader

    list_of_items_encoded=[]
    list_of_outcome_encoded=[]
    for i in range(len(dic_students)):

        c,d=preprocess_learning_trace(dic_students[i].learning_trace,corpus.nb_items)
        list_of_items_encoded.append(c)
        list_of_outcome_encoded.append(d)
    X=torch.stack(list_of_items_encoded)
    y=torch.stack(list_of_outcome_encoded)    
    dataset = TensorDataset(X, y)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion=nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in dataloader:
        # Forward pass
            predictions,_ = model(batch_X)
            
            # Compute loss
            loss = criterion(predictions, batch_y)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")


    item_hidden_states = {i: [] for i in range(corpus.nb_items)}
    item_outcomes = {i: [] for i in range(corpus.nb_items)}
    for i in range(len(dic_students)):
        hidden_states = model.rnn(list_of_items_encoded[i])[0]
        for t, (item_id, correctness) in enumerate(dic_students[i].learning_trace):
            item_hidden_states[item_id].append(hidden_states[t].detach().numpy())
            item_outcomes[item_id].append(correctness)



    trained_model=copy.deepcopy(model)


    return trained_model, item_hidden_states, item_outcomes

def get_fitted_models(classifier,item_hidden_states, item_outcomes):
    """
    Fit the classifier on the item hidden states and outcomes.
    
    Args:
        classifier (sklearn classifier): Classifier to fit.
        item_hidden_states (dict): Hidden states for each item.
        item_outcomes (dict): Outcomes for each item.
    
    Returns:
        dict: Fitted classifiers for each item.
    """
    fitted_models = {}
    for item_id in item_hidden_states:
        X = item_hidden_states[item_id]
        y = item_outcomes[item_id]
        
        fitted_models[item_id] = classifier.fit(X, y)
    return fitted_models