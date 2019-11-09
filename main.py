from tensorboardX import SummaryWriter
import os
import torch
import torchvision
import torch.nn as nn
from datetime import datetime

from torch.utils.data import DataLoader
from torch.autograd import Variable

from LipReadDataTrain import ReadData
from LipNet import LipNet, LipSeqLoss


def main():
    tot_iter = 0
    writer = SummaryWriter()
    train_image_file = os.path.join(os.path.abspath('.'), "data/lip_train")
    train_label_file = os.path.join(os.path.abspath('.'), "data/lip_train.txt")
    training_dataset = ReadData(train_image_file, train_label_file, seq_max_lens=24)
    training_data_loader = DataLoader(training_dataset, batch_size=20, shuffle=True, num_workers=12, drop_last=True)

    # GPU
    device = torch.device('cuda:0')
    # # CPU
    # device = torch.device('cpu')

    model = LipNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fc = LipSeqLoss().to(device)

    for epoch in range(1, 1000):
        model.train()
        for i_batch, sample_batched in enumerate(training_data_loader):
            tot_iter += 1
            input_data = Variable(sample_batched['volume']).to(device)
            labels = Variable(sample_batched['label']).to(device)
            length = Variable(sample_batched['length']).to(device)

            optimizer.zero_grad()
            result = model(input_data)
            loss = loss_fc(result, length, labels)
            loss.backward()
            optimizer.step()

            if tot_iter % 10 == 0:
                writer.add_scalar('train_loss', loss, tot_iter)
            # save model
        if epoch % 5 == 0:
            torch.save(model.state_dict(), "./weight/demo_net_epoch_{}.pt".format(epoch))



if __name__ == "__main__":
    main()
