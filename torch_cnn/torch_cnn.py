import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import gc



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output, x    # return x for visualization



def train(num_epochs, cnn, loaders):
    
    cnn.train()
        
    # Train the model
    total_step = len(loaders['train'])
    
    stats = gc.get_stats()
    initial_collected_gen_0 = stats[0]['collected']
    initial_collected_gen_1 = stats[1]['collected']
    initial_collected_gen_2 = stats[2]['collected']
    initial_collection_gen_0 = stats[0]['collections']
    initial_collection_gen_1 = stats[1]['collections']
    initial_collection_gen_2 = stats[2]['collections']
    # gc.enable()
    thresh = 0
    gc.set_threshold(thresh)
    
    with torch.profiler.profile(
        activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=0,
            warmup=0,
            active=100,
            repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/log_'+str(thresh)),
        with_stack=True
    ) as profiler:
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(loaders['train']):
                
                # gives batch data, normalize x when iterate train_loader
                b_x = Variable(images)   # batch x
                b_y = Variable(labels)   # batch y
                output = cnn(b_x)[0]               
                loss = loss_func(output, b_y)
                
                # clear gradients for this training step   
                optimizer.zero_grad()           
                
                # backpropagation, compute gradients 
                loss.backward()    
                # apply gradients             
                optimizer.step()
                profiler.step()
                # gc.collect()
                if (i+1) % 100 == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
                pass
            final_stats = gc.get_stats()
            average_collected_gen_0 = (final_stats[0]['collected'] - initial_collected_gen_0) / 6
            average_collected_gen_1 = (final_stats[1]['collected'] - initial_collected_gen_1) / 6
            average_collected_gen_2 = (final_stats[2]['collected'] - initial_collected_gen_2) / 6
            average_collection_gen_0 = (final_stats[0]['collections'] - initial_collection_gen_0) / 6
            average_collection_gen_1 = (final_stats[1]['collections'] - initial_collection_gen_1) / 6
            average_collection_gen_2 = (final_stats[2]['collections'] - initial_collection_gen_2) / 6
            print('average_collection_gen_0:', average_collection_gen_0)
            print('average_collection_gen_1:', average_collection_gen_1)
            print('average_collection_gen_2:', average_collection_gen_2)
            print('average_collected_gen_0:', average_collected_gen_0)
            print('average_collected_gen_1:', average_collected_gen_1)
            print('average_collected_gen_2:', average_collected_gen_2)
            pass
    
    
    pass


cnn = CNN()
loss_func = nn.CrossEntropyLoss() 
optimizer = optim.Adam(cnn.parameters(), lr = 0.01)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_data = datasets.MNIST(
    root = 'data',
    train = True,                         
    transform = ToTensor(), 
    download = True,            
)
test_data = datasets.MNIST(
    root = 'data', 
    train = False, 
    transform = ToTensor()
)

loaders = {
    'train' : torch.utils.data.DataLoader(train_data, 
                                          batch_size=100, 
                                          shuffle=True, 
                                          num_workers=1),
    
    'test'  : torch.utils.data.DataLoader(test_data, 
                                          batch_size=100, 
                                          shuffle=True, 
                                          num_workers=1),
}
num_epochs = 1
train(num_epochs, cnn, loaders)