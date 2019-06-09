# import libraries
import torch
import argparse
from PIL import Image
import numpy as np
from torchvision import models,transforms
from torch import nn, optim
import json

# get input from command line
parser = argparse.ArgumentParser()
parser.add_argument('image_path', action='store',
                    help='Store image path')

parser.add_argument('checkpoint', action='store',
                    help='Store checkpoint path')

parser.add_argument('--top_k', action='store',
                    dest='top_k',
                    type=int,
                    default = 5,
                    help='Store number of top probabilities returned')
 
parser.add_argument('--category_names', action='store',
                    dest='category_names',
                    default = 'cat_to_name.json',
                    help='Store mapping from category to name')

parser.add_argument('--gpu', action='store_true',
                    dest='use_gpu',
                    default = False,
                    help='Set to use GPU')
results = parser.parse_args()

# Load the checkpoint
def load_checkpoint(filepath):
    # load checkpoint file
    checkpoint = torch.load(filepath) 
    # initialize the model
    if checkpoint['model']=='vgg13':
        model = models.vgg13(pretrained=True)
    else:
        model = models.vgg19(pretrained=True)
    # define classifier
    model.classifier=checkpoint['classifier']
    # store mapping from class to index
    model.class_to_idx=checkpoint['class_idx']
    # load model state 
    model.load_state_dict(checkpoint['model_state_dict'])
    # define optimizer and load state
    optimizer = optim.Adam(model.classifier.parameters(), lr=checkpoint['learn_rate'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #retrieve epochs
    epochs = checkpoint['epochs']
    
    return model

# preprocess image for prediction
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Process a PIL image for use in a PyTorch model
    im = Image.open(image)
    #resize image
    im.thumbnail([256,256])
    #center crop image and normalize
    transform = transforms.CenterCrop(224)
    im = transform(im)
    # convert color channels
    np_image = np.asarray(im)/256
    # normalize numpy array
    np_image = (np_image-[0.485, 0.456, 0.406])/ [0.229, 0.224, 0.225]
    # change color chanel to first dimension
    np_image = np_image.transpose(2,0,1) 
    #create tensor
    image_tensor = torch.from_numpy(np_image)
    
    return image_tensor

# predict probabilities and classes for image
def predict(image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image=process_image(image_path)
    # without adding this, python gives warning "expected stride to be a single integer value or a list of 1 values 
    # to match the convolution dimensions, but got stride=[1, 1]". So I googled and get this solution
    # https://discuss.pytorch.org/t/expected-stride-to-be-a-single-integer-value-or-a-list/17612
    image.unsqueeze_(0)
    model.eval()
    with torch.no_grad():
        image = image.type(torch.FloatTensor)
        if results.use_gpu:
            image=image.cuda()
        logps = model.forward(image)
        ps = torch.exp(logps)
        top_p, top_idx = ps.topk(topk)
        class_idx =model.class_to_idx
        idx_class= dict(map(reversed, class_idx.items()))
        top_class = [idx_class[i] for i in top_idx.tolist()[0]]
    return top_p.tolist()[0],top_class

# build model from checkpoint
model = load_checkpoint(results.checkpoint)

if results.use_gpu:
    model.to('cuda')
else:
    model.to('cpu')

probs, classes = predict(results.image_path, model, results.top_k)

# mapping from class to names
with open(results.category_names, 'r') as f:
    cat_to_name = json.load(f)
cat = [cat_to_name[i] for i in classes]

print(probs)
print(cat)