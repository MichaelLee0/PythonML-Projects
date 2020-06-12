print("\n")     #Just so it's readable in the terminal
import torch 
import numpy
import torchvision # Contains datasets
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

# Import the datasets, "where its saved", "train ou non?", download: put this dataset in root dir?, transform gives us what transformations and the compose attribute lets us chain transforms together",

train = datasets.MNIST("", train = True, download = True, transform = transforms.Compose([transforms.ToTensor()]))      
test = datasets.MNIST("", train = False, download = True, transform = transforms.Compose([transforms.ToTensor()]))

# A data load pbject I guess seperates out our larger dataset into manageable sizes

trainset = torch.utils.data.DataLoader(train, batch_size = 10, shuffle = True)
test = torch.utils.data.DataLoader(test, batch_size = 10, shuffle = True)

# dataiter = iter(trainset)
# images, labels = dataiter.next()
# print(images.shape)
# print(labels.shape)
# plt.imshow(images[1].view(28,28), cmap= 'Greys_r')
# plt.show()


import torch.nn as nn                           # This holds objects that do the sameish? things as below. You INITIALIZE these
import torch.nn.functional as F                   # This apparently is more "functions"

class Net(nn.Module):                           # We want to inherit the methods of nn.Module in our new class "Net" so we write it as like a "base input" (i guess?)
    def __init__(self):                         # Just realized __init__ is like code for "blank space method" and declares starting variables and everything
        super().__init__()                      # super() in front of __init__() initializes the PARENT function (so we actually get its methods) and by virtue of being in an __init__ when called ALSO initializes the current one (which we need to initialize the other methods we're going to write)
        self.fc1 = nn.Linear(28*28, 64)         # this is initializing the first "fully connected (fc)" column of nodes (i think) and specifying it's input (flattened 28x28 MNIST pixel grid) (input, output) -> to 64 other nodes in the hidden layer
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)  
        self.fc4 = nn.Linear(64, 10)            # Outputs our 10 classes

    def forward(self, x):                       # Here we're just taking x, some input and passing it through first the fc1, then fc2, fc3 and so on
        x = F.relu(self.fc1(x))                 # F.relu comes from out torch "functions" package. it stands for rectifided linear. It is our ACTIVATION FUNCTION and tells us whether it is activated or not (tries to keep things from exploding)

        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))                 # Our activation function is working on the BACKEND/Outputs not the feeding side
        x = self.fc4(x)

        return F.log_softmax(x, dim = 1)        # dimension is which dimension do you want to normalize accross? Remember in python 1 == normal 2 so this is the 2nd dimension (accross the column). Remember we're passing in columns and our final layer IS a column vector


net = Net()         # Initializing an instance of class(Net) as "net"
# print(net)

# X = torch.rand(28,28)      # Example of passing in data and getting the output layer
# X = X.view(1,28*28)        # The one in there is how pyTorch likes it. It tells it to be prepared for any additional amount of data to be passed through (ie: this is only the FIRST, be prepared for a second, third, etc. (even when we don't have any more))
# output = net(X)
# print(output)

import torch.optim as optim

optimizer = optim.Adam(net.parameters(), lr = 0.001)        # Adam is just one of many optimization algo's it's very fast and efficient with memory. We pass in parameters, here ALL OF THEM lol. lr is our "learning rate" or our step in the direction of the gradient 
EPOCHS = 3              # How many FULL passes we will we make through our data

# net.train()             # Enables Droupout and batch normalization       
for epoch in range(EPOCHS):         # Trains our data with 3 rounds of 10 shuffled pictures and labels
    for data in trainset:           # This simply unpacks our training set data
        # data is a batch of feature sets (sets of data that describe our features) and labels
        X, y = data
        net.zero_grad()             # This is "resetting" the gradient between batches
        output = net(X.view(-1, 28*28))             # Passing data into the network. Resizing the data to be flat (-1 is arbitrary. I think you can iterate up through n for each data set?)
        loss = F.nll_loss(output, y)                # nll_loss is like logrithmic error
        loss.backward()                 # Backpropogate 
        optimizer.step()                # This is what does our adjustment
    print(loss)

print(output)

# How correct were we?
correct = 0 
total = 0

# net.eval()          # Disables Dropout and batch normalization
with torch.no_grad():

    for data in trainset:           # Its like data grabs an instance of 10 pictures with their labels
        X,y = data      # X has ALL 10 pictures and y has ALL TEN corresponding labels
        output = net(X.view(-1, 784))                # we reshape them to pass through the neural net and collect the output! the -1 says "we don't care about the x. YOU MAKE IT so that it have 784 columns, don't care how many different rounds there are"

        for idx, i in enumerate(output):             # idx is the "number address of each element", i contains the key for the dictionary
            if torch.argmax(i) == y[idx]:            # if the max arguement of the 
                correct += 1
            total += 1

print("Accuracy: ", round(correct/total, 3))

plt.imshow(X[0].view(28,28))             # Use matlibplot to plot out what our image data looks like (held onto after the loop)
plt.show

print(torch.argmax(net(X[0].view(-1,784))))         # Print what the highest scored prediction was when we pass in that data






print("\n")