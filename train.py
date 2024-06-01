
import torch
import torchvision as tv
import numpy as np
import torch.nn as nn
import argparse
from tqdm import tqdm
from tensorboardX import SummaryWriter
from models import *
    
def main(args):
    device = args.device
    
    name = args.best_ckpt_path.split('/')[-1].split('.')[0]
    model = get_resnet_through_name(name).to(device)
    
    transform = tv.transforms.Compose([tv.transforms.ToTensor(), \
                                        tv.transforms.Resize([112, 112])])


    train_data = tv.datasets.ImageFolder(args.data_path+'train/', transform = transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                            drop_last=True)
    
    
    test_data = tv.datasets.ImageFolder(args.data_path+'test/', transform = transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size,shuffle=True,drop_last=True)
      

    # optimizer = torch.optim.Adamax(model.parameters(), lr=args.lr, weight_decay=1e-5)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_function = torch.nn.NLLLoss()
    # loss_function = torch.nn.CrossEntropyLoss()
    
    best_loss = 1e9
    writer = SummaryWriter('logs/'+name)#runs
    for epoch in range(args.epochs):
        def solve(data_loader, evaluating = False):
            total_loss,true_pred = 0,0
            max_iter = len(data_loader)
            _iter = iter(data_loader)
            for i in tqdm(range(max_iter)):
                try:
                    x, y = next(_iter)
                except StopIteration:
                    break
                optimizer.zero_grad()
                if evaluating:
                    torch.no_grad()
                pred_y = model.forward(x.to(device))
                
                loss = loss_function(pred_y, y.to(device))
                
                true_pred += (pred_y.argmax(-1)==y.to(device)).sum(-1).item() 

                total_loss += loss.item()
                if not evaluating:
                    if ~(torch.isnan(loss) | torch.isinf(loss)):
                        loss.backward()
                        optimizer.step()

                if evaluating:
                    torch.enable_grad()
            return total_loss / max_iter , true_pred / max_iter / args.batch_size
        
        loss, true_pred = solve(train_loader)
        writer.add_scalar("train_loss",loss,epoch)
        writer.add_scalar("train_true",true_pred,epoch)
        print(f'training on epoch: {epoch} with loss {loss}...')
        print(f'training on epoch: {epoch} with pred {true_pred}...')
            
        loss, true_pred = solve(test_loader, evaluating=True)
        writer.add_scalar("val_loss",loss,epoch)
        writer.add_scalar("val_true",true_pred,epoch)
        print(f'testing on epoch: {epoch} with loss {loss}...')
        print(f'testing on epoch: {epoch} with pred {true_pred}...')
        
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), args.best_ckpt_path)
            
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Path settings
    parser.add_argument('--data_path', type=str, 
                        default='data/')
    parser.add_argument('--best_ckpt_path', type=str, 
                        default='ckpt/resnet50.pth')
    
    # Run settings
    parser.add_argument('--device', type=str, default='cuda:2')
    parser.add_argument('--batch_size', type=int, default=64)

    # Training settings
    parser.add_argument('--lr', type=float, default=6e-5)
    parser.add_argument('--epochs', type=int, default=100)
    
    
    args = parser.parse_args()
    print(str(args))
    main(args)