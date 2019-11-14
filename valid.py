import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from LipReadDataTest import ReadData as ReadDataTest
import opt


def valid(model, epoch, dataloader):
    n_samples = len(dataloader.dataset)
    # GPU
    device = torch.device('cuda:0')
    # # CPU
    # device = torch.device('cpu')
    model.eval()

    with torch.no_grad():
        running_corrects = 0.

        for i_batch, sample_batched in enumerate(dataloader):

            input_data = Variable(sample_batched['volume']).to(device)
            labels = Variable(sample_batched['label']).long().to(device)
            length = Variable(sample_batched['length']).to(device)

            outputs = model(input_data)

            batch_correct = validator(outputs, length, labels, every_frame=False)
            running_corrects += batch_correct

        acc = float(running_corrects) / n_samples
        print(f'Epoch:\t{epoch}\tAcc:{acc}\n')
        return acc


def validator(modelOutput, length, labels, every_frame=False):
    labels = labels.cpu()
    averageEnergies = torch.zeros((modelOutput.size(0), modelOutput.size(-1)))
    for i in range(modelOutput.size(0)):
        if every_frame:
            averageEnergies[i] = torch.mean(modelOutput[i, :length[i]], 0)
        else:
            averageEnergies[i] = modelOutput[i, length[i] - 1]

    _, maxindices = torch.max(averageEnergies, 1)
    count = torch.sum(maxindices == labels)
    return count

