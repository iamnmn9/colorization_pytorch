import torch
import os
from basic_model import Net


# from colorize_data import ColorizeData
# from basic_model import Net 
import torch.nn as nn

class Trainer:
    def __init__(self):
        pass
        # Define hparams here or load them from a config file
        self.lr = 0.001
        self.model = Net()
        self.epochs = 30
        self.batch_size = 32
        self.criterion = nn.MSELoss()
        #optimizer = torch.optim.Adam(model.parameters(),lr=0.01,weight_decay=0)
        #model = Net()
        # print('hi1')

    def train(self):
        # print('sdsdf')
        pass

        
        star = os.path.join('/content/images/train', "*.jpg")
        star = glob.glob(star)
        
        # print(star[0].shape)
        # print(len(star)) #3862

                
        star1 = os.path.join('/content/images/val', "*.jpg")
        star1 = glob.glob(star1)
        # print(len(star1)) #3862

            # train_dataset = ColorizeData('/content/images/train') #dataloarders
        train_dataset = ColorizeData(star)
        # print(train_dataset)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

        # print(train_dataloader)
        # val_dataset = ColorizeData(star1)
        # val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=True)

        
        # print('hi2')
        # Model
        # model = Net()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model=self.model.to(device)
        #model.train()
        # Just right before the actual usage
        
        # model = model.to(device)

        # Loss function to use
        criterion = nn.MSELoss()



        # You may also use a combination of more than one loss function 
        # or create your own.
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.0)  #Momentum=0.9
        # scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9, last_epoch=-1, verbose=True)    
            

            
        # train loop
        for epoch in range(self.epochs):
          self.model.train()
          # print("model enter")
          for input_transform,target_transform in train_dataloader:
                      inputs = torch.tensor(input_transform).to(device)
                      target = torch.tensor(target_transform).to(device)
                      inputs,target = inputs.cuda(),target.cuda()
                      # print("CHECKKKK")
                      # print(target.shape)

                      output_img = self.model(inputs)
                      # print(output_img.shape)
                      loss=criterion(output_img,target)
                      optimizer.zero_grad()
                      loss.backward()
                      optimizer.step()
                      #if input_transform % 50==0:
                        #%print("-------------Epoch ",epochs,"---------Loss:",loss.item(),"-------------Accuracy:", ((torch.sum(labels==torch.argmax(out, axis=1))/len(labels)).item())*100)
                      #sys#.stdout.flush()
                      
                      print(loss)
                      torch.cuda.empty_cache()
        # torch.save(self.model.state_dict(), '/content/mdoel/')
        




        


    def validate(self):
    #     model = Net()

        
        # Validation loop begin
        model = self.model

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model=model.to(device)

        # model.eval()

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)  #Momentum=0.9
        # scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9, last_epoch=-1, verbose=True)    
        star1 = os.path.join('/content/images/val', "*.jpg")
        star1 = glob.glob(star1)
        # print(len(star1)) #3862
        val_dataset = ColorizeData(star1)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=True)

        

        for epoch in range(self.epochs):
          model.eval()
          # print("model enter")
          for input_transform,target_transform in val_dataloader:
            inputs = torch.tensor(input_transform).to(device)
            target = torch.tensor(target_transform).to(device)
            inputs,target = inputs.cuda(),target.cuda()
            
            output_img = model(inputs)
            # print("validation final output check")
            print(output_img.shape)
            loss=criterion(output_img,target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        



        torch.save(self.model.state_dict(), '/content/model/model.pth')

        # img = Image.open('/content/images/val/1025.jpg')
        
        # transform = T.Compose([T.ToTensor(),
        #                                   T.Resize(size=(256,256)),
        #                                   T.Grayscale(),
        #                                   T.Normalize((0.5), (0.5))
        #                                   ]) 

def main():
  foo = Trainer()
  foo.train()
  foo.validate()

if __name__=='__main__':
  main()