import torch
import pandas as pd
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models.LipNet import LipNet
from LipReadDataTest import ReadData as ReadDataTest
import opt


def predict():
    test_dataset = ReadDataTest(opt.test_image_file, seq_max_lens=24)
    test_data_loader = DataLoader(test_dataset, batch_size=opt.num_workers, shuffle=True, num_workers=opt.num_workers, drop_last=False)

    # GPU
    device = torch.device('cuda:0')
    # # CPU
    # device = torch.device('cpu')

    model = LipNet().to(device)
    if len(opt.load_model):
        model.load_state_dict(torch.load("./weight/demo_net_epoch_2.pt"))
    model.eval()

    with torch.no_grad():
        col_key = []
        col_pre = []
        for i_batch, sample_batched in enumerate(test_data_loader):

            input_data = Variable(sample_batched['volume']).to(device)
            length = Variable(sample_batched['length']).to(device)

            keys = [i.split('/')[-1] for i in sample_batched['key']]

            outputs = model(input_data)
            average_volumns = torch.sum(outputs.data, 1)
            for i in range(outputs.size(0)):
                average_volumns[i] = outputs[i, :length[i]].sum(0)
            _, max_indexs = torch.max(average_volumns, 1)
            max_indexs = max_indexs.cpu().numpy().tolist()

            col_key += keys
            col_pre += max_indexs

    dictionary = pd.read_csv(opt.dictionary, encoding='utf8')
    word_list = dictionary.dict.tolist()
    character_label = [word_list[i] for i in col_pre]
    predict = pd.DataFrame([col_key, character_label]).T
    predict.to_csv('预测结果.csv',encoding='utf8', index=None, header=None)


if __name__ == "__main__":
    predict()
