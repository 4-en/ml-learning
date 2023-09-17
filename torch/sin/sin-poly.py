# estimate sin(x) using a polynomial neural network

import torch

x = torch.linspace(-2*3.1416, 2*3.1416, 1000)
y = torch.sin(x)


class SinPoly(torch.nn.Module):
    def __init__(self, degree=10):
        super(SinPoly, self).__init__()
        
        self.degree = degree

        self.weights = torch.nn.Parameter(torch.randn(degree))

    def forward(self, x):
        exponents = torch.arange(self.degree).float()
        x = x.flatten().unsqueeze(1)

        y = x ** exponents
        y = y * self.weights

        y = y.sum(dim=1)
        print(y.shape)
        return y.flatten()
    

model = SinPoly(degree=10)
model.train()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

EPOCHS = 100
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch} Loss: {loss.item()}")


model.eval()

while True:
    x = float(input("Enter a value for x: "))
    y = model(torch.tensor([x]))
    y_real = torch.sin(torch.tensor([x]))
    print(f"sin({x}) = {y.item()} (real: {y_real.item()})")

