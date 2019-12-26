
from FaceLandmarks_Network import *
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data.sampler import SubsetRandomSampler
def do_main():
    parser = argparse.ArgumentParser(description='Detector')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',				
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',		
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='save the current Model')
    parser.add_argument('--save-directory', type=str, default='trained_models',
                        help='learnt models are saving here')
    parser.add_argument('--phase', type=str, default='Train',   # Train/train, Predict/predict, Finetune/finetune
                        help='training, predicting or finetuning')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    print('loading data set......')
    train_set,test_set = get_train_test_set()
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=args.batch_size,shuffle=True)
    valid_loader = torch.utils.data.DataLoader(test_set,batch_size=args.test_batch_size)

    print('building model......')
    model = Net().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters,lr = args.lr,momentum=args.momentum)

    if args.phase == 'train' or args.phase=='Train':
        print('start train......')
        train_loss,valid_loss=train(args,train_loader,valid_loader,model,criterion,optimizer,device)
    else if args.phase = 'test' or args.phase=='Test':
        print('start test .......')

    elif args.phase == 'Finetune' or args.phase == 'finetune':
        print('===> Finetune')

        # how to do finetune?
    elif args.phase == 'Predict' or args.phase == 'predict':
        print('===> Predict')
        # how to do predict?
        

if __name__ == "__main__":
    do_main()




    


    