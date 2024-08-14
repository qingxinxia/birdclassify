import torch.nn as nn

class CNN_AE_encoder(nn.Module):
    def __init__(self, n_channels, n_classes, out_channels=128, backbone=True):
        super(CNN_AE_encoder, self).__init__()

        self.n_channels = n_channels * 2
        # self.datalen = 180  #args.len_sw
        # self.n_classes = n_classes   # check if correct a

        self.linear = nn.Linear(n_channels, self.n_channels)
        kernel_size = 5
        self.e_conv1 = nn.Sequential(nn.Conv2d(self.n_channels, 32,
                                               (1, kernel_size), bias=False,
                                               padding=(0, kernel_size // 2)),
                                     nn.BatchNorm2d(32),
                                     nn.Tanh())  # Tanh is MoIL paper
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0, return_indices=True)
        self.dropout = nn.Dropout(0.35)  # probability of samples to be zero

        self.e_conv2 = nn.Sequential(nn.Conv2d(32, 64,
                                               (1, kernel_size), bias=False,
                                               padding=(0, kernel_size // 2)),
                                     nn.BatchNorm2d(64),
                                     nn.Tanh())
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=2, padding=0, return_indices=True)

        self.e_conv3 = nn.Sequential(nn.Conv2d(64, out_channels,
                                               (1, kernel_size), bias=False,
                                               padding=(0, kernel_size // 2)),
                                     nn.BatchNorm2d(out_channels),
                                     nn.PReLU())
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0, return_indices=True)

        self.out_samples = 25
        self.out_dim = out_channels

        return

    def forward(self, x_input):  # x(batch,len180,dim6)
        x = self.linear(x_input)
        x = x.unsqueeze(2).permute(0, 3, 2, 1)  # outx(batch,dim,1,len)
        x1 = self.e_conv1(x)  # x1(batch,64,1,180)
        x1 = x1.squeeze(2)  # batch,32,180
        x, indice1 = self.pool1(x1)  # (batch,32,90)len减半,最后一维maxpool
        x = x.unsqueeze(2)
        x = self.dropout(x)
        # ---------
        x2 = self.e_conv2(x)  # batch,64,90
        x2 = x2.squeeze(2)
        x, indice2 = self.pool2(x2)
        x = x.unsqueeze(2)  # batch,64,45
        x = self.dropout(x)
        # ---------
        x3 = self.e_conv3(x)  # batch,128,45
        x3 = x3.squeeze(2)
        x_encoded, indice3 = self.pool3(x3)  # xencoded(batch,128,15)
        # x_encoded # batch,128,15
        return x_encoded, [indice1, indice2, indice3]


class CNN_AE_decoder(nn.Module):
    def __init__(self, n_channels, n_classes, out_channels=128, backbone=True):
        super(CNN_AE_decoder, self).__init__()

        self.n_channels = n_channels
        kernel_size = 5
        self.unpool1 = nn.MaxUnpool1d(kernel_size=2, stride=2, padding=0)
        self.d_conv1 = nn.Sequential(nn.ConvTranspose2d(out_channels, 64,
                                                        kernel_size=(1, kernel_size),
                                                        bias=False,
                                                        padding=(0, kernel_size // 2)),
                                     nn.BatchNorm2d(64),
                                     nn.Tanh())

        self.unpool2 = nn.MaxUnpool1d(kernel_size=4, stride=2, padding=0)
        self.d_conv2 = nn.Sequential(nn.ConvTranspose2d(64, 32,
                                                        kernel_size=(1, kernel_size),
                                                        stride=1, bias=False,
                                                        padding=(0, kernel_size // 2)),
                                     nn.BatchNorm2d(32),
                                     nn.PReLU())

        self.unpool3 = nn.MaxUnpool1d(kernel_size=2, stride=2, padding=0)
        self.d_conv3 = nn.Sequential(nn.ConvTranspose2d(32, n_channels,
                                                        kernel_size=(1, kernel_size),
                                                        stride=1, bias=False,
                                                        padding=(0, kernel_size // 2)),
                                     nn.BatchNorm2d(n_channels),
                                     nn.PReLU())

        self.linear = nn.Linear(n_channels, 3)
        if n_channels == 3:  # acc,gyro, where data length is 90
            self.reshapel = nn.Linear(89, 90)
        else:
            self.reshapel = nn.Linear(29, 30)
        return

    def forward(self, x_encoded, encode_indices):  # x_encoded(batch, 128, 25)
        x = self.unpool1(x_encoded, encode_indices[-1])  # out(batch, 64, 47)
        x = x.unsqueeze(2)
        x = self.d_conv1(x)  # out(batch, 128, 45)
        x = x.squeeze(2)
        # x = self.lin1(x)
        # ---------
        x = self.unpool2(x, encode_indices[-2])  # out(batch, 64, 90)
        x = x.unsqueeze(2)
        x = self.d_conv2(x)  # out(batch, 32, 91)
        x = x.squeeze(2)
        # ---------
        x = self.unpool3(x, encode_indices[0])  # x_decoded(batch,32,180)
        x = x.unsqueeze(2)
        x_decoded = self.d_conv3(x)
        x_decoded = x_decoded.squeeze(2)  # batch, 6, 180 = AE input
        # x_decoded = self.reshapel(x_decoded)
        # x_decoded = self.linear(x_decoded)
        return x_decoded


class CNN_AE(nn.Module):
    def __init__(self, n_channels, out_channels=128):
        super(CNN_AE, self).__init__()

        # self.backbone = backbone
        self.n_channels = n_channels  # input data dimension

        self.lin2 = nn.Identity()
        # self.out_dim = 25 * out_channels

        n_classes = 5  # not used
        self.encoder = CNN_AE_encoder(n_channels, n_classes,
                                      out_channels=out_channels, backbone=True)
        self.decoder = CNN_AE_decoder(n_channels, n_classes,
                                      out_channels=out_channels, backbone=True)

        # # if backbone == False:
        # self.classifier = self.encoder.classifier
        # # self.out_dim

        return

    def forward(self, x):  # x(batch, len180, dim6)
        x_encoded, encode_indices = self.encoder(x)  # x_encoded(batch, 128, 25)
        # todo, encoder output 改成batch,dim
        decod_out = self.decoder(x_encoded, encode_indices)  # x_decoded(batch, 6, 179)

        x_decoded = decod_out.permute(0, 2, 1)
        # x_decoded(batch, 180, 6), x_encoded(batch, 128, 15)
        return x_encoded, x_decoded
