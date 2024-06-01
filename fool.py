
import torch
import torchvision as tv
import numpy as np
import argparse,pickle
from tqdm import tqdm
from models import *
from tensorboardX import SummaryWriter
from diffusers import DDPMScheduler, UNet2DModel
from models import *

        

def main(args):
    device = args.device
    name = args.best_ckpt_path.split('/')[-1].split('.')[0]
    cnn_model = get_resnet_through_name(name).to(device)
    state_dict = torch.load(args.best_ckpt_path, map_location=torch.device('cpu'))
    cnn_model.load_state_dict(state_dict)
    cnn_model.eval()
    transform = tv.transforms.Compose([tv.transforms.ToTensor(), \
                                        tv.transforms.Resize([112,112])])


    
    test_data = tv.datasets.ImageFolder(args.data_path, transform = transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size)
    
    
    def loss_function(rebuilt_x, x, desire=0.5):
        variant_loss = torch.norm(x-rebuilt_x,p=2).sum(-1).sum(-1).sum(-1)
        confusion_loss = torch.tensor([0])
        confusion_loss = ((cnn_model(rebuilt_x.to(device),need_log=False)[:,0] - desire) ** 2).cpu()
        return confusion_loss + args.W * variant_loss, confusion_loss, args.W * variant_loss
    writer = SummaryWriter()#runs

    TIME_WASTE = torch.tensor([0]).to(device)
    for _epoch in range(1):
        def solve(data_loader, evaluating = False):
            y_output = []
            ccc = 0
            total_loss = 0
            max_iter = len(data_loader)
            _iter = iter(data_loader)
            for i in tqdm(range(max_iter)):
                try:
                    x, y = next(_iter)
                except StopIteration:
                    break
                y_ = y.item()
                desire = 0.4+y_*0.2
                model = UNETModel().to(device)
                optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-5)
                
                for epoch in tqdm(range(args.epochs)):
                    optimizer.zero_grad()
                    rebuilt_x = model(x.to(device),TIME_WASTE).sample.cpu()
                    loss,confusion_loss,variant_loss = loss_function(rebuilt_x, x, desire)
                    if ~(torch.isnan(loss) | torch.isinf(loss)):
                        loss.backward()
                        optimizer.step()
                        
                    writer.add_scalar(f"train_loss_{i}",loss.item(),epoch)
                    writer.add_scalar(f"train_confusion_loss_{i}",confusion_loss.item(),epoch)
                    writer.add_scalar(f"train_variant_loss_{i}",variant_loss.item(),epoch)
                    if(epoch == args.epochs-1):
                        total_loss+=loss.item()
                        for j in range(args.batch_size):
                            id = i * args.batch_size + j
                            tv.utils.save_image(rebuilt_x[j].cpu(),f'{args.output}{y_}/{id}.png')
                            tv.utils.save_image(x[j].cpu(),f'{args.output_true}{y_}/{id}.png')
                print(f"train_loss for {i}.png: ",loss.item())
                print(f"train_confusion_loss for {i}.png: ",confusion_loss.item())
                print(f"train_varient_loss for {i}.png: ",variant_loss.item())
                ccc += variant_loss.item()

            print(ccc)

            return total_loss / max_iter
        

        print(f'testing on epoch: {_epoch}...')
        loss = solve(test_loader, evaluating=True)        
        print(f'testing on epoch: {_epoch} with loss{loss}...')                        

        
            
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Path settings
    parser.add_argument('--data_path', type=str, 
                        default='data_2bfaked/')
    parser.add_argument('--best_ckpt_path', type=str, 
                        default='ckpt/resnet50.pth')
    parser.add_argument('--output', type=str, 
                        default='fake_data/')
    parser.add_argument('--output_true', type=str, 
                        default='true_data/')
    
    # Run settings
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--batch_size', type=int, default=1)

    # Training settings
    parser.add_argument('--lr', type=float, default=4e-3)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--W', type=float, default=4e-3)
    
    args = parser.parse_args()
    print(str(args))
    main(args)