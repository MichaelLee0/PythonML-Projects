print("\n")     #Just so it's readable in the terminal
import torch 
import torchvision # Contains datasets
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

# Import the datasets, "where its saved", "train ou non?", download: put this dataset in root dir?, transform gives us what transformations and the compose attribute lets us chain transforms together",

train = datasets.MNIST("", train = True, download = True, transform = transforms.Compose([transforms.ToTensor()]))      
test = datasets.MNIST("", train = False, download = True, transform = transforms.Compose([transforms.ToTensor()]))


# A data load pbject I guess seperates out our larger dataset into manageable sizes


trainset = torch.utils.data.DataLoader(train, batch_size = 10, shuffle = True)
test = torch.utils.data.DataLoader(test, batch_size = 10, shuffle = True)

# How we iterate through and print out our data

for data in trainset:
    X,y = data 

# print(len(X))
print(len(data[0]))
# print(len(y))

# print(y)


# # for data in trainset:
# #     print(data)
# #     break

# # Just want to verify our understanding of the data
# # So from the first ARRAY (it's an array) in data ^, take the first arary there. from the second element (vector of "solutions") take the first "solution" there 

# x, y = data[0][0], data[1][0]
# plt.imshow(data[0][0].view(28,28))
# plt.show()
# print(data[0][0].shape)

# # How to check for balance in a data set (make sure it doesn't become overly confident in one behavior)

# total = 0
# counter_dict = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}

# # My greasy first try way

# for data in trainset:
#     X, Y = data
#     total = range(len(X))
#     for n in total:
#         counter_dict[int(Y[int(n)])] += 1

# # A more slick method

# for data in trainset:
#     X,Y = data  #So each element of our dataset is comprised of two other elements (The pixel values and the key). We seperate each of those into X and Y
#     for y in Y:     # Then, for each element in our key
#         counter_dict[int(y)] += 1       # add one count to the corresponding "key group". You have to change from a char to a int if you want to iterate.
#         total += 1


# # # Print it out so it looks nice
# for i in counter_dict:
#     print(f"{i}: {counter_dict[i] / total * 100}")      # f print treats anything you put within the {} as a function


print("\n")

