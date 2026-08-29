# Behavior Cloning

Behavior cloning treats imitation learning as a supervised learning problem — directly learning a state-to-action mapping from expert demonstrations.

## Contents

- [1. Principle](#1-principle)
- [2. Simple Implementation](#2-simple-implementation)
- [3. Regularized Behavior Cloning](#3-regularized-behavior-cloning)

---

## 1. Principle

Behavior cloning converts imitation learning into a supervised learning problem:

$$
\min_\theta \sum_{(s,a) \in D} \mathcal{L}(\pi_\theta(s), a)
$$

**Advantages**:
- Simple to implement
- Sample efficient
- No reward design needed

**Main limitation — distribution shift**: during training, the policy only sees states visited by the expert; at deployment, small errors compound and push the robot into unseen states, where the policy fails.

---

## 2. Simple Implementation

```python
import torch
import torch.nn as nn

class BehaviorCloning:
    """
    Behavior cloning: the simplest imitation learning method
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)
        
    def forward(self, state):
        """Forward pass"""
        return self.policy(state)
    
    def train_step(self, states, actions):
        """Single training step"""
        # Predict actions
        pred_actions = self.policy(states)
        
        # MSE loss
        loss = nn.MSELoss()(pred_actions, actions)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, demonstrations, epochs=100, batch_size=32):
        """
        Train.
        
        demonstrations: {
            'states': [N, state_dim],
            'actions': [N, action_dim]
        }
        """
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(demonstrations['states']),
            torch.FloatTensor(demonstrations['actions'])
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        for epoch in range(epochs):
            total_loss = 0
            for states, actions in dataloader:
                loss = self.train_step(states, actions)
                total_loss += loss
                
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")
```

---

## 3. Regularized Behavior Cloning

```python
class RegularizedBC:
    """
    Regularized behavior cloning
    - Dropout regularization
    - L2 regularization
    - Perturbation regularization
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3, weight_decay=1e-4)
        
    def train_step(self, states, actions, noise_std=0.1):
        """Adversarial training with noise"""
        # Add noise for adversarial training
        noise = torch.randn_like(actions) * noise_std
        noisy_actions = actions + noise
        
        # Predict
        pred_actions = self.policy(states)
        
        # Standard BC loss
        bc_loss = nn.MSELoss()(pred_actions, actions)
        
        # Adversarial loss (against the noisy version)
        noisy_pred = self.policy(states)
        adv_loss = nn.MSELoss()(noisy_pred, noisy_actions)
        
        # Total loss
        loss = bc_loss + 0.1 * adv_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
```

---

## References

1. Pomerleau, D. A. (1989). ALVINN: An autonomous land vehicle in a neural network. NeurIPS.
2. Ross, S., Gordon, G., & Bagnell, D. (2011). A reduction of imitation learning to no-regret online learning. AISTATS.

---

*This chapter is continuously updated...*
