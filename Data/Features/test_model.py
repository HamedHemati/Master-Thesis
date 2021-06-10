import torchvision.models as models
import torch
import torch.nn as nn
from torch.autograd import Variable


path = '/home/hamed/Downloads/Pre-trained Models/vgg19-dcbb9e9d.pth'
vgg19 = models.vgg19(pretrained=False)
vgg19.load_state_dict(torch.load(path))
modified_classifier = nn.Sequential(*list(vgg19.classifier.children())[0:2])
vgg19.classifier = modified_classifier
print(vgg19)
inp = Variable(torch.randn(2, 3, 224, 224))
out = vgg19(inp)
print(out)











'''
path = '/home/hamed/Downloads/Pre-trained Models/resnet18-5c106cde.pth'
resnet_18 = models.resnet18(pretrained=False)
resnet_18.load_state_dict(torch.load(path))
model = nn.Sequential(*list(resnet_18.children())[:-1])
'''

'''
path = '/home/hamed/Downloads/Pre-trained Models/inception_v3_google-1a9a5a14.pth'
inception = models.inception_v3(pretrained=False, aux_logits=False)
dc = torch.load(path)
del(dc['AuxLogits.conv0.conv.weight'])
del(dc['AuxLogits.conv0.bn.weight'])
del(dc['AuxLogits.conv0.bn.bias'])
del(dc['AuxLogits.conv0.bn.running_mean'])
del(dc['AuxLogits.conv0.bn.running_var'])
del(dc['AuxLogits.conv1.conv.weight'])
del(dc['AuxLogits.conv1.bn.weight'])
del(dc['AuxLogits.conv1.bn.bias'])
del(dc['AuxLogits.conv1.bn.running_mean'])
del(dc['AuxLogits.conv1.bn.running_var'])
del(dc['AuxLogits.fc.weight'])
del(dc['AuxLogits.fc.bias'])
inception.load_state_dict(dc)
model = nn.Sequential(*list(inception.children())[:-1]) 
print(model)
'''
