import torch.nn as nn
import torch as torch

def main():

    # N=Batch size, C=Channels, D=Number of images, H=Height, W=Width
    input = torch.randn(12, 8, 8)  # C,H,W
#    print (input)
    print (input.shape)
    print("Conv1")
    m = nn.Conv2d(12, 64, 3, stride=1, padding=1)
    output = m(input)
#    print(output)
    print("m output")
    print(output.shape) # 24*12*8*8

    print("Conv2")
    m2 = nn.Conv2d(64, 192, 3, stride=1, padding=1)
    output = m2(output)
    print(output.shape)  # 24*768

    print("Conv3")
    m3 = nn.Conv2d(192,384,3,stride=1,padding=1)
    output = m3(output)
    print(output.shape)  # 24*768

    print("Conv4")
    m4 = nn.Conv2d(384,256,3,stride=1,padding=1)
    output = m4(output)
    print(output.shape)  # 24*768

    print("Conv5")
    m5 = nn.Conv2d(256,256,3,stride=1,padding=1)
    output = m5(output)
    print(output.shape)  # 24*768

    print("Flatten")
    output = torch.flatten(output, 0)
    print(output.shape)

    # Classification
    print("Linear1")
    m6 = nn.Linear(256*8*8, 4096)
    output = m6(output)
    print(output.shape)  # 24*768

    # Classification
    print("Linear2")
    m7 = nn.Linear(4096,1)
    output = m7(output)
    print(output.shape)  # 24*768
    print(output)





if __name__ == '__main__':
    main()

 # With square kernels and equal stride
# m = nn.Conv3d(16, 33, 3, stride=2)
# non-square kernels and unequal stride and with padding
# m = nn.Conv3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0))
# input = torch.randn(20, 16, 10, 50, 100)
# output = m(input)

   #
   # def __init__(self):
   #      super(ChessmaitCnn1,self).__init__()
   #      self.features = nn.Sequential(
   #          nn.Conv2d(12,64,kernel_size=3,padding=1),
   #          nn.ReLU(),
   #          nn.Conv2d(64,192,kernel_size=3,padding=1),
   #          nn.ReLU(),
   #          nn.Conv2d(192,384,kernel_size=3,padding=1),
   #          nn.ReLU(),
   #          nn.Conv2d(384,256,kernel_size=3,padding=1),
   #          nn.ReLU(),
   #          nn.Conv2d(256,256,kernel_size=3,padding=1),
   #          nn.ReLU(),
   #      )
   #
   #      self.classifier = nn.Sequential(
   #          nn.Dropout(),
   #          nn.Linear(256 * 6 * 6,4096),
   #          nn.ReLU(),
   #          nn.Dropout(),
   #          nn.Linear(4096,4096),
   #          nn.ReLU(),
   #          nn.Linear(4096,1)
   #      )
   #
   #  def forward(self,x):
   #      x = self.features(x)
   #      x = torch.flatten(x,1)
   #      x = self.classifier(x)
   #      return x