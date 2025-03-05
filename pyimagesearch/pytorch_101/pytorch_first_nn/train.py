import pyimagesearch.pytorch_101.pytorch_first_nn.mlp as mlp
from torch.optim import SGD
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
import torch.nn as nn
import torch

def next_batch(inputs, targets, batchSize):
    # loop over the dataset
    for i in range(0, inputs.shape[0], batchSize):
        # yield a tuple of the current batched data and labels
        yield (inputs[i:i+batchSize], targets[i:i+batchSize])
        
# specify our batch size, number of epochs, and learning rate
BATCH_SIZE=64
EPOCHS=10
LR=1e-2

# determine the device will be using for training
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("[INFO] training using {}...".format(DEVICE))

# generate a 3-class classification problem with 1000 data points
# where each data point is a 4D feature vector
print("[INFO] preparing data...")
(X, y) = make_blobs(n_samples=1000, n_features=4, centers=3, 
                    cluster_std=2.5, random_state=95)
trainX, testX, trainY, testY = train_test_split(X, y,
                                                test_size=0.15, random_state=95)
trainX = torch.from_numpy(trainX).float()
testX = torch.from_numpy(testX).float()
trainY = torch.from_numpy(trainY).float()
testY = torch.from_numpy(testY).float()

# initialize our model and show the architecture
mlp_model = mlp.get_training_model().to(DEVICE)
print(mlp_model)

# create optimizer and loss function
opt = SGD(mlp_model.parameters(), lr=LR)
lossFunc = nn.CrossEntropyLoss()
    
# create a template to summerize current training progress
trainTemplate = "epoch: {} train loss: {:.3f} train accuracy: {:.3f}"

# loop through epochs
for epoch in range(0, EPOCHS):
    # initialize tracker variables and set our model to trainable
    print("[INFO] epoch: {}...".format(epoch))
    trainLoss = 0
    trainAcc = 0
    samples = 0
    mlp_model.train()
    
    # loop over the current batch of data
    for (batchX, batchY) in next_batch(trainX, trainY, BATCH_SIZE):
        # flash data to the current device, run it through our
        # model, and calculate loss
        (batchX, batchY) = (batchX.to(DEVICE), batchY.to(DEVICE))
        predictions = mlp_model(batchX)
        loss = lossFunc(predictions, batchY.long())
        
        # zeros the gradients accumulate from the previous steps
        # perform backpropagation, and update model parameters
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        # update the training loss, accuracy, the number of samples visited
        trainLoss += loss.item()*batchY.size(0)
        trainAcc += (predictions.max(1)[1]==batchY).sum().item()
        samples += batchY.size(0)
        
    # display the training progress on the current training batch
    print(trainTemplate.format(EPOCHS+1, trainLoss/samples, trainAcc/samples))
    
    # initialize tracker variables for testing, then set our model to 
    # evaluation mode
    testLoss = 0
    testAcc = 0
    samples = 0
    mlp_model.eval()
    
    # initialize a no-gradient context
    with torch.no_grad():
        # loop over the current batch of test data;
        for (batchX, batchY) in next_batch(testX, testY, BATCH_SIZE):
            # flash the data to the current device
            (batchX, batchY) = (batchX.to(DEVICE), batchY.to(DEVICE))

			# run data through our model and calculate loss
            predictions = mlp_model(batchX)
            loss = lossFunc(predictions, batchY.long())
            
            # update the test loss, accuracy, and the number of sample visited
            testLoss += loss.item() * batchY.size(0)
            testAcc += (predictions.max(1)[1]==batchY).sum().item()
            samples += batchY.size(0)
            
        # display model progress on the current test batch
        testTemplate = "epoch: {} test loss: {:.3f} test accuracy: {:.3f}"
        print(testTemplate.format(EPOCHS+1, (testLoss/samples), (testAcc/samples)))
        