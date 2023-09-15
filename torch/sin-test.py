import torch
from torch import nn

import math
import random

LOAD_MODEL = False
MODEL_NAME = 'sin-model.pth'

class SinNet(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.hidden = nn.Linear(1, hidden_size)

        #self.hidden2 = nn.Linear(20, hidden_size)

        self.output = nn.Linear(hidden_size, 1)

        #self.seq = nn.Sequential(
        #    nn.Linear(1, 20),
        #    nn.ReLU(),
        #    nn.Linear(20, hidden_size),
        #    nn.ReLU(),
        #    nn.Linear(hidden_size, 1)
        #)

    def forward(self, x):
        x = self.hidden(x)
        x = torch.relu(x)
        #x = self.hidden2(x)
        #x = torch.relu(x)
        x = self.output(x)
        return x
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
# device name
# print(torch.cuda.get_device_name(0))
    
model = SinNet(20)
if LOAD_MODEL:
    model.load_state_dict(torch.load(MODEL_NAME))
model = model.to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

EPOCHS = 30
BATCH_SIZE = 1000

model.train()

for epoch in range(EPOCHS):
    for _ in range(BATCH_SIZE):
        x = random.random() * math.pi * 2
        y = math.sin(x)
        x = torch.tensor([x], device=device)
        y = torch.tensor([y], device=device)

        optimizer.zero_grad()

        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()

        optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}: {loss.item()}')

# test sin

model.eval()
while True:
    x = input('x: ')
    if x == 'q':
        break
    x = float(x) % (math.pi*2)
    actual = math.sin(x)
    x = torch.tensor([x], device=device)

    y_pred = model(x)
    print('Actual:', actual)
    print('Predicted:', y_pred.item())

# save model
torch.save(model.state_dict(), MODEL_NAME)