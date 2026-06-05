import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import CIFAR10 as cif
from torchvision import transforms
import torch
from torch.utils.data import DataLoader

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])
test_transform=transforms.Compose([
    transforms.ToTensor()]
)
train_data = cif(
    root="./data",
    train=True,
    download=True,
    transform=train_transform
)
test_data = cif(
    root ="./data",
    train=False,
    download=True,
    transform=test_transform
)
train_loader = DataLoader(
    train_data,
    batch_size=32,
    shuffle=True
)
test_loader = DataLoader(
    test_data,
    batch_size=32,
    shuffle=False
)

image,label=next(iter(train_loader)) 
image1,label1=next(iter(test_loader))

class ResBlock(nn.Module):

  def __init__ (self,in_channels,out_channels,stride):

    super().__init__()
    self.c1=nn.Conv2d(in_channels,out_channels,3,padding=1)
    self.b1=nn.BatchNorm2d(out_channels)
    self.r=nn.ReLU()
    self.c2=nn.Conv2d(out_channels,out_channels,3,padding=1)
    self.b2=nn.BatchNorm2d(out_channels)
    self.s=nn.Conv2d(in_channels,out_channels,1,stride=stride)
    self.b=nn.BatchNorm2d(out_channels)
    self.p=nn.MaxPool2d(2)
    self.stride=stride
    self.out=out_channels
    self.inc=in_channels
  def forward(self,x):

    if (self.stride==1 and self.out==self.inc):  #To skip the projection shortcut if dimensions are same
      skip=x
    else:
      skip=self.s(x)
      skip=self.b(skip)
    out=self.c1(x)
    out=self.b1(out)
    out=self.r(out)
    if (self.stride==2):  
      out=self.p(out)  #Pooling to match dimensions
    out=self.c2(out)
    out=self.b2(out)
    out+=skip
    out=self.r(out)
    return out

class Mycnn(nn.Module):

    def __init__ (self):
      super().__init__()
      self.c1=nn.Conv2d(3,64,3,padding=1)
      self.b1=nn.BatchNorm2d(64)
      self.r=nn.ReLU()
      self.res1=ResBlock(64,64,1)
      self.res2=ResBlock(64,64,1)
      self.pool=nn.MaxPool2d(2)
      self.res3=ResBlock(64,128,2)
      self.res4=ResBlock(128,128,1)
      self.res5=ResBlock(128,256,2)
      self.res6=ResBlock(256,256,1)
      self.res7=ResBlock(256,512,2)
      self.res8=ResBlock(512,512,1)
      self.l1=nn.Linear(512,512)
      self.l2=nn.Linear(512,10)
      self.d=nn.Dropout(0.2)
      self.f=nn.Flatten()
      self.a=nn.AdaptiveAvgPool2d(1)
    def forward(self,x):
        out=self.c1(x)
        out=self.b1(out)
        out=self.r(out)
        out=self.res1(out)
        out=self.res2(out)
        out=self.res3(out)
        out=self.res4(out)
        out=self.res5(out)
        out=self.res6(out)
        out=self.res7(out)
        out=self.res8(out)
        out=self.a(out)
        out=self.f(out)
        out=self.l1(out)
        out=self.r(out)
        out=self.d(out)
        out=self.l2(out)
        return out
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Mycnn().to(device)
optimizer=optim.Adam(model.parameters(),lr=0.001,weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=15,gamma=0.1)
criterion=nn.CrossEntropyLoss()

for i in range(50):
  avg_loss=0
  model.train()
  for x,y in train_loader:
    x=x.to(device)
    y=y.to(device)
    optimizer.zero_grad()
    output=model(x)
    loss=criterion(output,y)
    loss.backward()
    optimizer.step()
    avg_loss+=loss.item()
  scheduler.step()
  if i%5==0:
       print(f"EPOCH: {i}|| AVG LOSS: {avg_loss/len(train_loader)}")
with torch.no_grad():
    model.eval()
    acc1=0
    total=0
    acc2=0
    for xt,yt in train_loader:
        xt=xt.to(device)
        yt=yt.to(device)
        y_pred=model(xt)
        pred=torch.argmax(y_pred,dim=1)
        acc1+=(pred==yt).sum().item()
        total+=yt.size(0)
    print(f"TRAIN_ACC: {100*acc1/total} ")
    total=0
    for x,y in test_loader:
        x=x.to(device)
        y=y.to(device)
        y_pred=model(x)
        pred=torch.argmax(y_pred,dim=1)
        acc2+=(pred==y).sum().item()
        total+=y.size(0)
    print(f"TEST_ACC: {100*acc2/total} ")