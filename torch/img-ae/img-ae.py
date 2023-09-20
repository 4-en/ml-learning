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

TARGET_DIM = 3
# create model
model = Autoencoder(layers=5, input_dim=28*28, target_dim=TARGET_DIM)
model.to(device)

# create optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# create loss function
loss_fn = nn.MSELoss()

# create data loader
data_loader = torch.utils.data.DataLoader(img, batch_size=64, shuffle=True)

TRAIN = True

if TRAIN:
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
else:
    # load model
    model.load_state_dict(torch.load("model.pth"))


model.eval()

from PIL import Image, ImageSequence

#scale images
FACTOR = 4
#output = output.repeat(FACTOR, axis=1).repeat(FACTOR, axis=2)
#output = output.reshape(-1, 28*FACTOR)


# save as gif
# images = []
# for i in range(output.shape[0]):
#     img = Image.fromarray(output[i,:,:])
#     images.append(img)

# images[0].save('img-ae.gif', save_all=True, append_images=images[1:], duration=100, loop=0)



# save model
torch.save(model.state_dict(), "model.pth")


# show image

from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QSlider, QMainWindow
from PyQt5.QtGui import QPixmap

app = QApplication([])

window = QMainWindow()
window.resize(800, 800)
window.setWindowTitle("Digit decoder")

layout = QVBoxLayout()
widget = QWidget()
window.setCentralWidget(widget)
widget.setLayout(layout)

label = QLabel("Hello World")
layout.addWidget(label)

# show image in qt
from PIL import ImageQt

values = [0.0 for i in range(TARGET_DIM)]
def regenerate():
    t = torch.tensor([values]).to(device)
    output = model.decode(t)
    output = output.view(-1, 28, 28) * 255.0
    output = output.round().to(torch.uint8).cpu().numpy()

    img = Image.fromarray(output[0,:,:])
    img = img.resize((img.width*FACTOR, img.height*FACTOR))
    img = ImageQt.ImageQt(img)
    pixmap = QPixmap.fromImage(img)
    # scale by 4
    pixmap = pixmap.scaledToWidth(pixmap.width()*4)
    label.setPixmap(pixmap)

regenerate()

def sliderCallback(id):
    def callback(value):
        value = value/100
        values[id] = value
        regenerate()
    
    return callback

for i in range(TARGET_DIM):
    slider = QSlider()
    slider.valueChanged.connect(sliderCallback(i))
    slider.setOrientation(1)
    slider.setMinimum(0)
    slider.setMaximum(100)
    slider.setValue(0)
    layout.addWidget(slider)


window.show()

app.exec_()


        