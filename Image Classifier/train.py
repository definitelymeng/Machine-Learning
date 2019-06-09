# import libraries
import argparse
import torch
from torchvision import datasets, transforms,models
from torch import nn, optim

# get input from command line
parser = argparse.ArgumentParser()
parser.add_argument('data_directory', action='store',
                    help='Store data directory')
 
parser.add_argument('--save_dir', action='store',
                    dest='save_dir',
                    default = 'checkpoint.pth',
                    help='Store checkpoint path')

parser.add_argument('--arch', action='store',
                    dest='model',
                    default = 'vgg19',
                    help='Store chosen model')

parser.add_argument('--learning_rate', action='store',
                    type=float,
                    dest='learning_rate',
                    default = 0.001,
                    help='Store learning rate')

parser.add_argument('--hidden_units', action='store',
                    type=int,
                    dest='hidden_units',
                    default = 1024,
                    help='Store hidden units')

parser.add_argument('--epochs', action='store',
                    type=int,
                    dest='epochs',
                    default = 5,
                    help='Store epochs')

parser.add_argument('--gpu', action='store_true',
                    dest='use_gpu',
                    default = False,
                    help='Set to use GPU')

results = parser.parse_args()

# load the data
data_dir = results.data_directory
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Define transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(40),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

# Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir,transform =train_transforms)
validation_data = datasets.ImageFolder(valid_dir,transform =test_transforms)
test_data = datasets.ImageFolder(test_dir,transform =test_transforms)

# Define dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(validation_data,batch_size=64)
testloader = torch.utils.data.DataLoader(test_data,batch_size=64)

# Building and training the classifier
if results.model == 'vgg13':
    model = models.vgg13(pretrained=True)
else:
    model = models.vgg19(pretrained=True)

# freeze model parameters
for param in model.parameters():
    param.requires_grad = False

# build own classifier
classifier = nn.Sequential(nn.Linear(25088, results.hidden_units),
                           nn.ReLU(),
                           nn.Dropout(0.2),
                           nn.Linear(results.hidden_units, 512),
                           nn.ReLU(),
                           nn.Dropout(0.2),
                           nn.Linear(512, 102),
                           nn.LogSoftmax(dim=1))

for param in model.parameters():
    param.requires_grad = False
    
classifier = nn.Sequential(nn.Linear(25088, 1024),
                           nn.ReLU(),
                           nn.Dropout(0.2),
                           nn.Linear(1024, 512),
                           nn.ReLU(),
                           nn.Dropout(0.2),
                           nn.Linear(512, 102),
                           nn.LogSoftmax(dim=1))

# assign new classifier to model
model.classifier =classifier

# using GPU mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if results.use_gpu:
    model.to(device)

# define loss function and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=results.learning_rate)

epochs = results.epochs
steps = 0
running_loss = 0
print_every = 20

#keep workspace alive
from workspace_utils import keep_awake

for i in keep_awake(range(epochs)):
    for images, labels in trainloader:
        if results.use_gpu:
            images, labels = images.to(device), labels.to(device)
        steps += 1
        
        optimizer.zero_grad()
        logps = model.forward(images)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss
        
        if steps % print_every == 0:
            valid_loss = 0
            valid_accuracy = 0
            model.eval()
            
            with torch.no_grad():
                for images, labels in validloader:
                    if results.use_gpu:
                        images, labels = images.to(device), labels.to(device)
                    logps = model.forward(images)
                    batch_loss = criterion(logps, labels)
                    
                    valid_loss += batch_loss
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    valid_accuracy += torch.mean(equals.type(torch.FloatTensor))
                    
            print(f"Epoch {i+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                  f"Validation accuracy: {valid_accuracy/len(validloader):.3f}")
            running_loss = 0
            model.train()

# Save the checkpoint 
# map classes to indices
model.class_to_idx = train_data.class_to_idx

#create checkpoint (learned from pytorch official tutorial)
# method source: https://pytorch.org/tutorials/beginner/saving_loading_models.html
torch.save({
            'epochs': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'class_idx':model.class_to_idx,
            'classifier':model.classifier,
            'model': results.model,
            'learn_rate': results.learning_rate,
            }, results.save_dir)