from tensorboardX import SummaryWriter
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from LipReadDataTrain import ReadData
from LipReadDataTest import ReadData as ReadDataTest
from models.LipNet import LipReading, NLLSequenceLoss
from valid import valid
import opt


def main():
    tot_iter = 0
    writer = SummaryWriter()
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    training_dataset = ReadData(opt.train_image_file, opt.train_label_file, seq_max_lens=opt.seq_max_lens)
    training_data_loader = DataLoader(training_dataset, batch_size=opt.batch_size,
                                      shuffle=True, num_workers=opt.num_workers, drop_last=True)
    valid_dataset = ReadData(opt.valid_image_file, opt.valid_label_file, seq_max_lens=opt.seq_max_lens)
    valid_data_loader = DataLoader(valid_dataset, batch_size=opt.batch_size, shuffle=True,
                                   num_workers=opt.num_workers, drop_last=False)

    device = torch.device('cuda:0')

    model = LipReading().to(device)
    if len(opt.load_model):
        print("[LOAD PRE-TRAINED MODEL]")
        model.load_state_dict(torch.load("./weight/demo_net_epoch_2.pt"))
    optimizer = torch.optim.Adadelta(model.parameters())
    loss_fc = NLLSequenceLoss().to(device)
    model.train()

    for epoch in range(1, opt.epoch):

        for i_batch, sample_batched in enumerate(training_data_loader):
            tot_iter += 1
            input_data = Variable(sample_batched['volume'], require_grad=True).to(device)
            label = Variable(sample_batched['label']).long().to(device)
            length = Variable(sample_batched['length']).to(device)

            optimizer.zero_grad()
            result = model(input_data)
            loss = loss_fc(result, length, label)
            loss.backward()
            optimizer.step()
            print(f"{tot_iter}:{loss}")
            if tot_iter % 10 == 0:
                writer.add_scalar('train_loss', loss, tot_iter)
            # save model
            torch.save(model.state_dict(), "./weight/demo_net_epoch_{}.pt".format(epoch))
        valid(model, epoch, valid_data_loader)


if __name__ == "__main__":
    main()
