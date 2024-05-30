import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from main.backbone import resnet

class CNNWrapper(nn.Module):
    def __init__(self, backbone, checkpoint_path):
        super(CNNWrapper, self).__init__()
        self.backbone = backbone
        self.model = self.initialize_backbone(checkpoint_path)

    def forward(self, x):
        with torch.no_grad():
            x = self.model(x)
        x = x.reshape(x.size(0), -1)
        return x

    def initialize_backbone(self, checkpoint_path):
        if self.backbone == 'resnet50':
            model = resnet.resnet50(pretrained=False)
            model.load_state_dict(torch.load(checkpoint_path))
        elif self.backbone == 'resnet101':
            model = resnet.resnet101(pretrained=False)
            model.load_state_dict(torch.load(checkpoint_path))
        elif self.backbone == 'resnet152':
            model = resnet.resnet152(pretrained=False)
            model.load_state_dict(torch.load(checkpoint_path))
        elif self.backbone == 'vgg16':
            model = models.vgg16(pretrained=True)
        elif self.backbone == 'vgg19':
            model = models.vgg19(pretrained=True)

        modules = list(model.children())[:-1]
        model = nn.Sequential(*modules)
        return model

class VideoEncoder(nn.Module):
    def __init__(self, in_size, units):
        super(VideoEncoder, self).__init__()
        self.linear = nn.Linear(in_size, units)
        self.lstm = nn.LSTM(units, units, batch_first=True)
        self.reset_parameters()

    def forward(self, Xv):
        Xv = self.linear(Xv)
        Xv = F.relu(Xv)
        Xv, (hi, ci) = self.lstm(Xv)
        Xv = Xv[:, -1, :]
        hi, ci = hi[0, :, :], ci[0, :, :]
        return Xv, (hi, ci)

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'hh' in name:
                    nn.init.orthogonal_(param.data)
                else:
                    nn.init.xavier_uniform_(param.data)
            else:
                nn.init.zeros_(param.data)

class CommandDecoder(nn.Module):
    def __init__(self, units, vocab_size, embed_dim, bias_vector=None):
        super(CommandDecoder, self).__init__()
        self.units = units
        self.embed_dim = embed_dim
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm_cell = nn.LSTMCell(embed_dim, units)
        self.logits = nn.Linear(units, vocab_size, bias=True)
        self.softmax = nn.LogSoftmax(dim=1)
        self.reset_parameters(bias_vector)

    def forward(self, Xs, states):
        Xs = self.embed(Xs)
        hi, ci = self.lstm_cell(Xs, states)
        x = self.logits(hi)
        x = self.softmax(x)
        return x, (hi, ci)

    def init_hidden(self, batch_size):
        h0 = torch.zeros(batch_size, self.units)
        c0 = torch.zeros(batch_size, self.units)
        return (h0, c0)

    def reset_parameters(self, bias_vector):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'hh' in name:
                    nn.init.orthogonal_(param.data)
                else:
                    nn.init.xavier_uniform_(param.data)
            else:
                nn.init.zeros_(param.data)
        nn.init.uniform_(self.embed.weight.data, -0.05, 0.05)
        if bias_vector is not None:
            self.logits.bias.data = torch.from_numpy(bias_vector).float()

class CommandLoss(nn.Module):
    def __init__(self, ignore_index=0):
        super(CommandLoss, self).__init__()
        self.cross_entropy = nn.NLLLoss(reduction='sum', ignore_index=ignore_index)

    def forward(self, input, target):
        return self.cross_entropy(input, target)

class Video2Command():
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    def build(self, bias_vector=None):
        self.video_encoder = VideoEncoder(in_size=list(self.config.BACKBONE.values())[0], units=self.config.UNITS)
        self.command_decoder = CommandDecoder(units=self.config.UNITS, vocab_size=self.config.VOCAB_SIZE, embed_dim=self.config.EMBED_SIZE, bias_vector=bias_vector)
        self.video_encoder.to(self.device)
        self.command_decoder.to(self.device)
        self.loss_objective = CommandLoss()
        self.loss_objective.to(self.device)
        self.params = list(self.video_encoder.parameters()) + list(self.command_decoder.parameters())
        self.optimizer = torch.optim.Adam(self.params, lr=self.config.LEARNING_RATE)
        if not os.path.exists(os.path.join(self.config.CHECKPOINT_PATH, 'saved')):
            os.makedirs(os.path.join(self.config.CHECKPOINT_PATH, 'saved'))

    def train(self, train_loader):
        def train_step(Xv, S):
            loss = 0.0
            self.optimizer.zero_grad()
            Xv, states = self.video_encoder(Xv)
            S_mask = S != 0
            for timestep in range(self.config.MAXLEN - 1):
                Xs = S[:, timestep]
                probs, states = self.command_decoder(Xs, states)
                loss += self.loss_objective(probs, S[:, timestep + 1])
            loss = loss / S_mask.sum()
            loss.backward()
            self.optimizer.step()
            return loss

        self.video_encoder.train()
        self.command_decoder.train()
        for epoch in range(self.config.NUM_EPOCHS):
            total_loss = 0.0
            for i, (Xv, S, clip_names) in enumerate(train_loader):
                Xv, S = Xv.to(self.device), S.to(self.device)
                loss = train_step(Xv, S)
                total_loss += loss
                if i % self.config.DISPLAY_EVERY == 0:
                    print('Epoch {}, Iter {}, Loss {:.6f}'.format(epoch + 1, i, loss))
            print('Total loss for epoch {}: {:.6f}'.format(epoch + 1, total_loss / (i + 1)))
            if (epoch + 1) % self.config.SAVE_EVERY == 0:
                self.save_weights(epoch + 1)
        return

    def evaluate(self, test_loader, vocab):
        assert self.config.MODE == 'test'
        y_pred, y_true = [], []
        for i, (Xv, S_true, clip_names) in enumerate(test_loader):
            Xv, S_true = Xv.to(self.device), S_true.to(self.device)
            S_pred = self.predict(Xv, vocab)
            y_pred.append(S_pred)
            y_true.append(S_true)
        y_pred = torch.cat(y_pred, dim=0)
        y_true = torch.cat(y_true, dim=0)
        return y_pred.cpu().numpy(), y_true.cpu().numpy()

    def predict(self, Xv, vocab):
        self.video_encoder.eval()
        self.command_decoder.eval()
        with torch.no_grad():
            S = torch.zeros((Xv.shape[0], self.config.MAXLEN), dtype=torch.long)
            S[:, 0] = vocab('<sos>')
            S = S.to(self.device)
            Xv, states = self.video_encoder(Xv)
            for timestep in range(self.config.MAXLEN - 1):
                Xs = S[:, timestep]
                probs, states = self.command_decoder(Xs, states)
                preds = torch.argmax(probs, dim=1)
                S[:, timestep + 1] = preds
        return S

    def save_weights(self, epoch):
        torch.save({
            'VideoEncoder_state_dict': self.video_encoder.state_dict(),
            'CommandDecoder_state_dict': self.command_decoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, os.path.join(self.config.CHECKPOINT_PATH, 'saved', 'v2c_epoch_{}.pth'.format(epoch)))
        print('Model saved.')

    def load_weights(self, save_path):
        print('Loading...')
        checkpoint = torch.load(save_path)
        self.video_encoder.load_state_dict(checkpoint['VideoEncoder_state_dict'])
        self.command_decoder.load_state_dict(checkpoint['CommandDecoder_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('Model loaded.')
