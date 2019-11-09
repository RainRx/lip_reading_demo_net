from tensorboardX import SummaryWriter
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from LipReadDataTrain import ReadData
from models.LipNet import LipNet, LipSeqLoss
from valid import valid
import opt


def main():
    tot_iter = 0
    writer = SummaryWriter()
    training_dataset = ReadData(opt.train_image_file, opt.train_label_file, seq_max_lens=opt.seq_max_lens)
    training_data_loader = DataLoader(training_dataset, batch_size=opt.batch_size,
                                      shuffle=True, num_workers=opt.num_workers, drop_last=True)

    # GPU
    device = torch.device('cuda:0')
    # # CPU
    # device = torch.device('cpu')

    model = LipNet().to(device)
    if len(opt.load_model):
        model.load_state_dict(torch.load("./weight/demo_net_epoch_2.pt"))
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)
    loss_fc = LipSeqLoss().to(device)

    for epoch in range(1, opt.epoch):
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
        valid(model, epoch)


if __name__ == "__main__":
    main()
