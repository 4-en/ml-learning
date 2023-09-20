# autoencoder to encode mnist image data

import torch
from torch import nn

class Autoencoder(nn.Module):
    def __init__(self, layers=3, input_dim=28*28, target_dim=1):
        super(Autoencoder, self).__init__()
        self.layers = layers
        self.target_dim = target_dim
        self.input_dim = input_dim
        
        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()

        for i in range(self.layers):
            if i == 0:
                self.encoder.add_module("encoder_{}".format(i), nn.Linear(self.input_dim, self.input_dim))
                self.encoder.add_module("encoder_relu_{}".format(i), nn.ReLU())

                self.decoder.add_module("decoder_{}".format(i), nn.Linear(self.target_dim, self.input_dim))
                self.decoder.add_module("decoder_relu_{}".format(i), nn.ReLU())
            elif i == self.layers-1:
                self.encoder.add_module("encoder_{}".format(i), nn.Linear(self.input_dim, self.target_dim))
                self.encoder.add_module("encoder_sig_{}".format(i), nn.Sigmoid())
                self.decoder.add_module("decoder_{}".format(i), nn.Linear(self.input_dim, self.input_dim))

            else:
                self.encoder.add_module("encoder_{}".format(i), nn.Linear(self.input_dim, self.input_dim))
                self.encoder.add_module("encoder_relu_{}".format(i), nn.ReLU())
                self.decoder.add_module("decoder_{}".format(i), nn.Linear(self.input_dim, self.input_dim))
                self.decoder.add_module("decoder_relu_{}".format(i), nn.ReLU())

       
    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def encode(self, x):
        x = self.encoder(x)
        return x
    
    def decode(self, x):
        x = self.decoder(x)
        return x
    

# load dataset
from torchvision import datasets, transforms

data = datasets.MNIST(root='data', train=True, download=True, transform=transforms.ToTensor())

img = data.data.view(-1, 28*28).float()
img = img/255.0


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


# create model
model = Autoencoder(layers=5, input_dim=28*28, target_dim=1)
model.to(device)

# create optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# create loss function
loss_fn = nn.MSELoss()

# create data loader
data_loader = torch.utils.data.DataLoader(img, batch_size=64, shuffle=True)

# train model
EPOCHS = 20
model.train()
for epoch in range(EPOCHS):
    for data in data_loader:
        data = data.to(device)
        output = model(data)
        loss = loss_fn(output, data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Epoch [{}/{}], Loss: {:.4f}".format(epoch+1, EPOCHS, loss.item()))


model.eval()
# generate images
values = torch.arange(0, 1, 1/64).to(device).view(-1, 1)
#values = torch.cat((values, values), dim=1)
output = model.decode(values)
output = output.view(-1, 28, 28) * 255.0
output = output.round().to(torch.uint8).cpu().numpy()

import PIL
from PIL import Image, ImageSequence

#scale images
FACTOR = 4
#output = output.repeat(FACTOR, axis=1).repeat(FACTOR, axis=2)
#output = output.reshape(-1, 28*FACTOR)


# save as gif
images = []
for i in range(output.shape[0]):
    img = Image.fromarray(output[i,:,:])
    images.append(img)

images[0].save('img-ae.gif', save_all=True, append_images=images[1:], duration=100, loop=0)



# save model
torch.save(model.state_dict(), "model.pth")



        