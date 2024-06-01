import torch
import torchvision as tv
import numpy as np
import argparse,pickle,math
from tqdm import tqdm
from models import *
from tensorboardX import SummaryWriter


def main(args):
    device = args.device
    name = args.best_ckpt_path.split('/')[-1].split('.')[0]
    model = get_resnet_through_name(name).to(device)
    model.eval()
    output = f'output/{name}_{args.data_path.replace("/","")}.txt'
    state_dict = torch.load(args.best_ckpt_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    
    transform = tv.transforms.Compose([tv.transforms.ToTensor(), \
                                        tv.transforms.Resize([112,112])])


    
    test_data = tv.datasets.ImageFolder(args.data_path, transform = transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size)
    loss_function = torch.nn.NLLLoss()
      

    for epoch in range(1):
        def solve(data_loader, evaluating = False):
            y_output = []
            total_loss = 0
            true_pred = 0
            # assuming 1 as P, 0 as N
            TP,FP,TN,FN = 0,0,0,0
            max_iter = len(data_loader)
            _iter = iter(data_loader)
            for i in tqdm(range(max_iter)):
                try:
                    x, y = next(_iter)
                except StopIteration:
                    break
                
                torch.no_grad()
                pred_y = model.forward(x.to(device))
                loss = loss_function(pred_y, y.to(device)).sum(-1)
                true_pred += (pred_y.argmax(-1)==y.to(device)).sum(-1).item() 
                
                if(pred_y.argmax(-1)==1 and y.item()==1):
                    TP+=1
                if(pred_y.argmax(-1)==1 and y.item()==0):
                    FP+=1
                if(pred_y.argmax(-1)==0 and y.item()==0):
                    TN+=1
                if(pred_y.argmax(-1)==0 and y.item()==1):
                    FN+=1
                    
                y_output += pred_y[:,0].detach().cpu().numpy().tolist()
                total_loss += loss.item()
            # print((TP+TN)/(TP+TN+FP+FN))
            # print((TP)/(TP+FP))
            # print((TN)/(TN+FN))
            # print((TP)/(TP+FN))
            # print((TN)/(TN+FP))
            return total_loss / max_iter , true_pred / max_iter / args.batch_size, y_output
        

        print(f'testing on epoch: {epoch}...')
        loss, pred, y_output = solve(test_loader, evaluating=True)            
        with open(output, "w+") as f:
            variance = 0
            for t in y_output:
                variance += (math.exp(t)-0.5)**2
                f.write(str(math.exp(t))+'\n')
                
            f.write(f'variance: {epoch} with loss {variance/len(y_output)}...\n')
            f.write(f'testing on epoch: {epoch} with loss {loss}...\n')
            f.write(f'testing on epoch: {epoch} with pred {pred}...\n')
            # pickle.dump(y_output, f)
        print(f'testing on epoch: {epoch} with loss {loss}...')
        print(f'testing on epoch: {epoch} with pred {pred}...')
        
        
            
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Path settings
    parser.add_argument('--data_path', type=str, 
                        default='data/')
    parser.add_argument('--best_ckpt_path', type=str, 
                        default='ckpt/resnet50.pth')

    
    # Run settings
    parser.add_argument('--device', type=str, default='cuda:3')
    parser.add_argument('--batch_size', type=int, default=1)

    
    
    args = parser.parse_args()
    print(str(args))
    main(args)