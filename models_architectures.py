from torch import nn

class ContextPredictor(nn.Module):

    '''
    Encoder-decoder for recreating images.
    '''

  def __init__(self, input_shape, hidden_units):
    super().__init__()

    #encoding layers

    self.conv_block_1 = nn.Sequential(
      nn.Conv2d(in_channels=input_shape,
                    out_channels=hidden_units,
                    kernel_size=3,
                    stride=1,
                    padding=1),
      nn.ReLU(),
      nn.Conv2d(in_channels=hidden_units,
                    out_channels=hidden_units,
                    kernel_size=3,
                    stride=1,
                    padding=1),
      nn.ReLU(),
      nn.Dropout(0.1),
      nn.MaxPool2d(kernel_size=2,
                      stride=2)
      )

    self.conv_block_2 = nn.Sequential(
      nn.Conv2d(in_channels=hidden_units,
                    out_channels=hidden_units,
                    kernel_size=3,
                    stride=1,
                    padding=1),
      nn.ReLU(),
      nn.Conv2d(in_channels=hidden_units,
                    out_channels=hidden_units,
                    kernel_size=3,
                    stride=1,
                    padding=1),
      nn.ReLU(),
      nn.Dropout(0.1),
      nn.MaxPool2d(kernel_size=2,
                      stride=2)
      )

    self.encoder = nn.Sequential(
      self.conv_block_1,
      self.conv_block_2
      )

    #decoding layers

    self.deconv_block_1 = nn.Sequential(
        nn.ConvTranspose2d(in_channels=hidden_units,
                          out_channels=hidden_units,
                          kernel_size=3,
                          stride=2,
                          padding=1,
                          output_padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(in_channels=hidden_units,
                          out_channels=hidden_units,
                          kernel_size=3,
                          stride=1,
                          padding=1),
        nn.ReLU(),
        nn.Dropout(0.1)
    )

    self.deconv_block_2 = nn.Sequential(
        nn.ConvTranspose2d(in_channels=hidden_units,
                          out_channels=hidden_units,
                          kernel_size=3,
                          stride=2,
                          padding=1,
                          output_padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(in_channels=hidden_units,
                          out_channels=input_shape,
                          kernel_size=3,
                          stride=1,
                          padding=1)
    )


    self.decoder = nn.Sequential(
      self.deconv_block_1,
      self.deconv_block_2
    )

  @staticmethod #static function, not instanction method
  def initialize_weights_xavier(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
      nn.init.xavier_uniform_(m.weight)
      if m.bias is not None: #if layer has a bias
        nn.init.zeros_(m.bias) #set bias to 0


  def forward(self, x):
    x = self.encoder(x)
    # print(x.shape)
    x = self.decoder(x)
    # print(x.shape)

    return x

class SEMNet(nn.Module):

'''
Model for prediction task.
'''

  def __init__(self, input_shape: int,
               hidden_units: int,
               output_shape: int):
    super().__init__()
    self.conv_block_1 = nn.Sequential(
        nn.Conv2d(in_channels=input_shape,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=0),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,
                     stride=2)
    )
    self.conv_block_2 = nn.Sequential(
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=0),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=0),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,
                     stride=2)
    )
    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=hidden_units*53*53,
                  out_features=output_shape)
    )

  def forward(self, x):
    x = self.conv_block_1(x)
    # print(x.shape)
    x = self.conv_block_2(x)
    # print(x.shape)
    x = self.classifier(x)
    # print(x.shape)
    return x
