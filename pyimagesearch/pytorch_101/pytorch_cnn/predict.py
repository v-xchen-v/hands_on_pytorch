# USAGE
# python predict.py --model output/model.pth



# set the numpy seed for better reproducibility
import numpy as np
np.random.seed(42)

# import the necessary packages
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision.transforms import ToTensor
from torchvision.datasets import KMNIST
import argparse
import imutils
import torch
import cv2
import matplotlib
matplotlib.use("Agg") # 'Agg' is for headless servers (does not open a window, only saves images).
import matplotlib.pyplot as plt

from flask import Flask, Response
app = Flask(__name__)
def show_image(image: np.array):
    _, buffer = cv2.imencode('.jpg', image)
    yield (b'--frame\r\n'
        b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/image')
def image_feed():
    return Response(show_image(), mimetype='image/jpg')

# construct the arguments parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True,
                help="path to the trained PyTorch nodel")
# TODO:?what's the vars() for?
args = vars(ap.parse_args())

# set the device we will be using to test the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the KMNIST dataset and randomly grab 10 data points
print("[INFO] loading the KMNIST test dataset...")
testData = KMNIST(root="data", train=False, download=True, 
                  transform=ToTensor())

idxs = np.random.choice(range(0, len(testData)), size=(10,))
testData = Subset(testData, idxs)

# initialize the test data loader
testDataLoader = DataLoader(testData, batch_size=1)

# load the model and set it to evaluation mode
model = torch.load(args["model"]).to(device)
model.eval()

# switch off autograd
# TODO: ?what will happen if not swith off autograd?
with torch.no_grad():
    # loop over the test set
    for (image, label) in testDataLoader:
        # grab the original image and ground truth label
        # TODO: explain the squeeze op
        origImage = image.numpy().squeeze(axis=(0,1))
        gtLabel=testData.dataset.classes[label.numpy()[0]]
        
        # sent the input to the device and make prediction on it
        image = image.to(device)
        pred = model(image)
        
        # find the class label index with the largest corresponding 
        # probability
        idx = pred.argmax(axis=1).cpu().numpy()[0]
        predLabel = testData.dataset.classes[idx]
        
        # convert the image from grayscale to RGB (so we can draw on 
        # it) and resize it (so we can more easiky see it on our screen)
        # TODO: ?dstack?
        origImage = np.dstack([origImage]*3)
        origImage = imutils.resize(origImage, width=128)
        
        # draw the predicted class label on it
        color = (0, 255, 0) if gtLabel == predLabel else (0, 0, 255)
        cv2.putText(origImage, gtLabel, (2, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.95, color, 2)
        
        # display the result in terminal and show the input image
        print("[INFO] ground truth label: {}, predicted label: {}".format(
            gtLabel, predLabel
        ))
        
        # cv2.imshow("image", origImage)
        # cv2.waitKey(0)
        # plt.imshow(origImage/255)
        # plt.axis("off")
        # plt.show()
        # show_image(origImage)
            # Display image using Matplotlib
        plt.imshow(origImage)
        plt.axis("off")
        image_path = "current_image.png"
        plt.savefig(image_path, bbox_inches="tight")  # Save instead of show

        # Print instruction and wait for key press
        input(f"Showing {image_path}. Press ENTER to show the next image...")
# if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=5000)