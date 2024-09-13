import helper_functions
import torch
import torch.nn as nn
import torch.optim as optim

STATE_SIZE = 11
Q_VALUE_SIZE = 3
LR = 1e-3

# Check if GPU is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()

        # Define layers
        self.fc1 = nn.Linear(STATE_SIZE, 128)  # Input layer with moderate number of neurons
        self.bn1 = nn.BatchNorm1d(128)  # Batch normalization layer
        self.relu1 = nn.ReLU()  # ReLU activation
        self.dropout1 = nn.Dropout(p=0.3)  # Dropout layer for regularization
        
        self.fc2 = nn.Linear(128, 256)  # Second hidden layer with more neurons
        self.bn2 = nn.BatchNorm1d(256)  # Batch normalization layer
        self.relu2 = nn.ReLU()  # ReLU activation
        self.dropout2 = nn.Dropout(p=0.3)  # Dropout layer for regularization

        self.fc3 = nn.Linear(256, 128)  # Third hidden layer with reduced neurons for a tapering effect
        self.bn3 = nn.BatchNorm1d(128)  # Batch normalization layer
        self.relu3 = nn.ReLU()  # ReLU activation
        self.dropout3 = nn.Dropout(p=0.3)  # Dropout layer for regularization
        
        self.fc4 = nn.Linear(128, 32)  # Fourth hidden layer
        self.bn4 = nn.BatchNorm1d(32)  # Batch normalization layer
        self.relu4 = nn.ReLU()  # ReLU activation
        self.dropout4 = nn.Dropout(p=0.3)  # Dropout layer for regularization

        self.output_layer = nn.Linear(32, Q_VALUE_SIZE)  # Output layer for Q-values

    def forward(self, x):
        # Forward pass through the network
        x = self.dropout1(self.relu1(self.bn1(self.fc1(x))))
        x = self.dropout2(self.relu2(self.bn2(self.fc2(x))))
        x = self.dropout3(self.relu3(self.bn3(self.fc3(x))))
        x = self.dropout4(self.relu4(self.bn4(self.fc4(x))))
        x = self.output_layer(x)
        return x

# Initialize Q-network and target Q-network
q_net = QNetwork()
target_q_net = QNetwork()

# Define the optimizer
optimizer = torch.optim.Adam(q_net.parameters(), lr=LR)

# Loss function
criterion = nn.MSELoss()

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

def compute_loss(experiences, gamma):
    states, actions, rewards, next_states, done_values = experiences
    
    max_target_qsa = target_q_net(next_states).max(dim=1)[0]
    
    target_q = done_values * rewards + (1 - done_values) * (rewards + gamma* max_target_qsa)
    
    predicted_q = q_net(states)
    action_indices = torch.argmax(actions, dim=1)  # Find the index of the maximum action for each batch element

    # Use gather to select the elements from predicted_q
    predicted_q = torch.gather(predicted_q, dim=1, index=action_indices.unsqueeze(1)).squeeze(1)
    
    loss = criterion(target_q, predicted_q)
    return loss

# Training function for one step
def agent_learn(experiences, gamma):
    q_net.train()
    loss = compute_loss(experiences, gamma)
    # Zero the gradients
    optimizer.zero_grad()
    # Backward pass to compute gradients
    loss.backward()
    # Apply gradients to update the network
    optimizer.step()
    # Soft update of the target Q-network
    helper_functions.target_qNet_softupdate(q_net, target_q_net)