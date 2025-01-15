import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

#####################
# load and prepare the data
#####################

# download training data from open datasets
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# download test data from open datasets
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

# create data loaders
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape}, {y.dtype}")
    break

#####################
# create the model
#####################

# check if a GPU is available
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# define the model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # flatten the image to a vector
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            # 1st fully connected layer 
            nn.Linear(28*28, 512),
            # activation function
            nn.ReLU(),
            # 2nd fully connected layer
            nn.Linear(512, 512),
            # activation function
            nn.ReLU(),
            # 3rd fully connected layer
            nn.Linear(512, 10)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

#####################
# optimizing the model parameters
#####################

# define the loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# define the training loop
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        # compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

# define the testing loop
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    # set the model to evaluation mode
    model.eval()
    test_loss, correct = 0, 0
    # turn off gradients
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            # compute prediction error
            pred = model(X)
            # compute the loss and transform the tensor to a scalar
            test_loss += loss_fn(pred, y).item()
            # compute the number of correct predictions
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")
    
# train the model
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    # train and test the model in every epoch
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)

#####################
# saving models
#####################

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

#####################
# loading models
#####################

model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth", weights_only=True))

# use the model to make predictions
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')