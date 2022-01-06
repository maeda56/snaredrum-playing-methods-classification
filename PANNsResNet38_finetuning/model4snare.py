from pickle import FALSE
from torch import nn
from torchsummary import summary
from models import *
import math

class my_ResNet38(nn.Module):
    def __init__(self,
                 sample_rate,
                 window_size,
                 hop_size,
                 mel_bins,
                 fmin,
                 fmax,
                 classes_num,
                 freeze_base):
        super(my_ResNet38, self).__init__()
        audioset_classes_num = 527

        self.base = ResNet38(sample_rate, window_size, hop_size, mel_bins, fmin, fmax, audioset_classes_num)
        self.fc_transfer = nn.Linear(2048, classes_num, bias = True)

        if freeze_base:     #base's weigh fix ?
            for param in self.base.parameters():
                param.requires_grad = False

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_transfer)

    def load_from_pretrain(self, pretrained_checkpoint_path):
        checkpoint = torch.load(pretrained_checkpoint_path, map_location=lambda storage, loc: storage) 
        self.base.load_state_dict(checkpoint['model'])

    def forward(self, input, mixup_lambda=None):#input:(batch_size, data_length)
        output_dict = self.base(input, mixup_lambda)
        embedding = output_dict['embedding']

        clipwise_output = torch.log_softmax(self.fc_transfer(embedding), dim=-1)
        output_dict['clipwise_output'] = clipwise_output

        return output_dict['clipwise_output']
