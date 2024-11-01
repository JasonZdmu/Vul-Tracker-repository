import math

from transformers import RobertaTokenizer, RobertaModel
import torch
from torch import nn as nn
import numpy as np
from torch.nn import functional as F
from torch.autograd import Variable
from torch.nn import TransformerEncoder, TransformerEncoderLayer, Transformer
import logging, os
import torch
import torch.nn as nn
import numpy as np
from transformers import RobertaModel


# 行级粒度
class PatchClassifier(nn.Module):
    def __init__(self):
        super(PatchClassifier, self).__init__()
        self.HIDDEN_DIM = 768
        self.before_linear = nn.Linear(5 * self.HIDDEN_DIM, self.HIDDEN_DIM)
        self.after_linear = nn.Linear(5 * self.HIDDEN_DIM, self.HIDDEN_DIM)
        self.combine = nn.Linear(2 * self.HIDDEN_DIM, self.HIDDEN_DIM)

        self.out_proj = nn.Linear(self.HIDDEN_DIM, 2)

    def forward(self, before_batch, after_batch):
        d1, d2, d3 = before_batch.shape
        before_batch = torch.reshape(before_batch, (d1, d2 * d3))
        after_batch = torch.reshape(after_batch, (d1, d2 * d3))

        before = self.before_linear(before_batch)
        after = self.after_linear(after_batch)
        combined = self.combine(torch.cat([before, after], axis=1))

        out = self.out_proj(combined)

        return out


class CnnClassifier(nn.Module):
    """An 1D Convulational Neural Network for Sentence Classification."""

    def __init__(self,
                 embed_dim=768,
                 filter_sizes=[2, 3, 4],
                 num_filters=[100, 100, 100],
                 num_classes=2,
                 dropout=0.5):
        super(CnnClassifier, self).__init__()
        self.embed_dim = embed_dim
        print(filter_sizes)
        # Conv Network
        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(in_channels=self.embed_dim,
                      out_channels=num_filters[i],
                      kernel_size=filter_sizes[i])
            for i in range(len(filter_sizes))
        ])
        # Fully-connected layer and Dropout
        self.fc = nn.Linear(2 * np.sum(num_filters), num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, before_batch, after_batch):
        """Perform a forward pass through the network.

        Args:
            input_ids (torch.Tensor): A tensor of token ids with shape
                (batch_size, max_sent_length)

        Returns:
            logits (torch.Tensor): Output logits with shape (batch_size,
                n_classes)
        """

        # Get embeddings from `input_ids`. Output shape: (b, max_len, embed_dim)
        x_embed_before = before_batch
        # batch, file, hidden_dim

        # Permute `x_embed` to match input shape requirement of `nn.Conv1d`.
        # Output shape: (b, embed_dim, max_len)
        x_reshaped_before = x_embed_before.permute(0, 2, 1)

        # Apply CNN and ReLU. Output shape: (b, num_filters[i], L_out)
        x_conv_list_before = [F.relu(conv1d(x_reshaped_before)) for conv1d in self.conv1d_list]

        # Max pooling. Output shape: (b, num_filters[i], 1)
        x_pool_list_before = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
                              for x_conv in x_conv_list_before]

        # Concatenate x_pool_list to feed the fully connected layer.
        # Output shape: (b, sum(num_filters))
        x_fc_before = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list_before],
                                dim=1)

        # # Compute logits. Output shape: (b, n_classes)
        # out = self.fc(self.dropout(x_fc_before))

        ############################################

        x_embed_after = after_batch

        # Permute `x_embed` to match input shape requirement of `nn.Conv1d`.
        # Output shape: (b, embed_dim, max_len)
        x_reshaped_after = x_embed_after.permute(0, 2, 1)

        # Apply CNN and ReLU. Output shape: (b, num_filters[i], L_out)
        x_conv_list_after = [F.relu(conv1d(x_reshaped_after)) for conv1d in self.conv1d_list]

        # Max pooling. Output shape: (b, num_filters[i], 1)
        x_pool_list_after = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
                             for x_conv in x_conv_list_after]

        # Concatenate x_pool_list to feed the fully connected layer.
        # Output shape: (b, sum(num_filters))
        x_fc_after = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list_after],
                               dim=1)

        # Compute logits. Output shape: (b, n_classes)

        x_fc = torch.cat([x_fc_before, x_fc_after], axis=1)
        out = self.fc(self.dropout(x_fc))

        return out


class VariantTwoClassifier(nn.Module):
    def __init__(self):
        super(VariantTwoClassifier, self).__init__()
        self.HIDDEN_DIM = 768
        self.HIDDEN_DIM_DROPOUT_PROB = 0.3

        self.linear = nn.Linear(5 * self.HIDDEN_DIM, self.HIDDEN_DIM)
        self.relu = nn.ReLU()
        self.drop_out = nn.Dropout(self.HIDDEN_DIM_DROPOUT_PROB)
        self.out_proj = nn.Linear(self.HIDDEN_DIM, 2)

    def forward(self, file_batch, need_final_feature=False):
        d1, d2, d3 = file_batch.shape
        file_batch = torch.reshape(file_batch, (d1, d2 * d3))

        commit_embedding = self.linear(file_batch)

        x = commit_embedding
        x = self.relu(x)
        final_feature = x

        x = self.drop_out(x)
        out = self.out_proj(x)

        if need_final_feature:
            return out, final_feature
        else:
            return out


class VariantTwoFineTuneClassifier(nn.Module):
    def __init__(self):
        super(VariantTwoFineTuneClassifier, self).__init__()
        self.HIDDEN_DIM = 768
        self.code_bert = RobertaModel.from_pretrained("microsoft/codebert-base", num_labels=2)
        self.classifier = VariantTwoClassifier()

    def forward(self, input_list_batch, mask_list_batch):
        d1, d2, d3 = input_list_batch.shape
        input_list_batch = torch.reshape(input_list_batch, (d1 * d2, d3))
        mask_list_batch = torch.reshape(mask_list_batch, (d1 * d2, d3))
        embeddings = self.code_bert(input_ids=input_list_batch, attention_mask=mask_list_batch).last_hidden_state[:, 0,
                     :]
        embeddings = torch.reshape(embeddings, (d1, d2, self.HIDDEN_DIM))

        out = self.classifier(embeddings)

        return out

    def freeze_codebert(self):
        if not isinstance(self, nn.DataParallel):
            for param in self.code_bert.parameters():
                param.requires_grad = False
        else:
            for param in self.module.code_bert.parameters():
                param.requires_grad = False


class VariantSixClassifier(nn.Module):
    def __init__(self):
        super(VariantSixClassifier, self).__init__()
        self.HIDDEN_DIM = 768
        self.HIDDEN_DIM_DROPOUT_PROB = 0.3

        self.before_linear = nn.Linear(5 * self.HIDDEN_DIM, self.HIDDEN_DIM)
        self.after_linear = nn.Linear(5 * self.HIDDEN_DIM, self.HIDDEN_DIM)
        self.combine = nn.Linear(2 * self.HIDDEN_DIM, self.HIDDEN_DIM)

        self.relu = nn.ReLU()
        self.drop_out = nn.Dropout(self.HIDDEN_DIM_DROPOUT_PROB)
        self.out_proj = nn.Linear(self.HIDDEN_DIM, 2)

    def forward(self, before_batch, after_batch, need_final_feature=False):
        d1, d2, d3 = before_batch.shape
        before_batch = torch.reshape(before_batch, (d1, d2 * d3))
        after_batch = torch.reshape(after_batch, (d1, d2 * d3))

        before = self.before_linear(before_batch)
        after = self.after_linear(after_batch)
        combined = self.combine(torch.cat([before, after], axis=1))

        x = combined
        x = self.relu(x)

        final_feature = x

        x = self.drop_out(x)
        out = self.out_proj(x)

        if need_final_feature:
            return out, final_feature
        else:
            return out


class VariantThreeClassifier(nn.Module):
    """An 1D Convulational Neural Network for Sentence Classification."""

    def __init__(self,
                 embed_dim=768,
                 filter_sizes=[3, 4, 5],
                 num_filters=[100, 100, 100],
                 num_classes=2,
                 dropout=0.5):
        super(VariantThreeClassifier, self).__init__()
        self.embed_dim = embed_dim
        print(filter_sizes)
        # Conv Network
        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(in_channels=self.embed_dim,
                      out_channels=num_filters[i],
                      kernel_size=filter_sizes[i])
            for i in range(len(filter_sizes))
        ])
        # Fully-connected layer and Dropout
        self.fc = nn.Linear(np.sum(num_filters), num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, code, need_final_feature=False):
        """Perform a forward pass through the network.

        Args:
            input_ids (torch.Tensor): A tensor of token ids with shape
                (batch_size, max_sent_length)

        Returns:
            logits (torch.Tensor): Output logits with shape (batch_size,
                n_classes)
        """

        # Get embeddings from `input_ids`. Output shape: (b, max_len, embed_dim)
        x_embed = code

        # Permute `x_embed` to match input shape requirement of `nn.Conv1d`.
        # Output shape: (b, embed_dim, max_len)
        x_reshaped = x_embed.permute(0, 2, 1)

        # Apply CNN and ReLU. Output shape: (b, num_filters[i], L_out)
        x_conv_list = [F.relu(conv1d(x_reshaped)) for conv1d in self.conv1d_list]

        # Max pooling. Output shape: (b, num_filters[i], 1)
        x_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
                       for x_conv in x_conv_list]

        # Concatenate x_pool_list to feed the fully connected layer.
        # Output shape: (b, sum(num_filters))
        x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list],
                         dim=1)

        # Compute logits. Output shape: (b, n_classes)
        final_feature = x_fc

        out = self.fc(self.dropout(x_fc))

        if need_final_feature:
            return out, final_feature
        else:
            return out


class VariantThreeFineTuneClassifier(nn.Module):
    def __init__(self):
        super(VariantThreeFineTuneClassifier, self).__init__()
        self.HIDDEN_DIM = 768
        self.code_bert = RobertaModel.from_pretrained("microsoft/codebert-base", num_labels=2)
        self.classifier = VariantThreeClassifier()

    def forward(self, input_list_batch, mask_list_batch):
        d1, d2, d3 = input_list_batch.shape
        input_list_batch = torch.reshape(input_list_batch, (d1 * d2, d3))
        mask_list_batch = torch.reshape(mask_list_batch, (d1 * d2, d3))
        embeddings = self.code_bert(input_ids=input_list_batch, attention_mask=mask_list_batch).last_hidden_state[:, 0,
                     :]
        embeddings = torch.reshape(embeddings, (d1, d2, self.HIDDEN_DIM))

        out = self.classifier(embeddings)

        return out

    def freeze_codebert(self):
        if not isinstance(self, nn.DataParallel):
            for param in self.code_bert.parameters():
                param.requires_grad = False
        else:
            for param in self.module.code_bert.parameters():
                param.requires_grad = False


class VariantThreeFineTuneOnlyClassifier(nn.Module):
    def __init__(self):
        super(VariantThreeFineTuneOnlyClassifier, self).__init__()
        self.HIDDEN_DIM = 768
        self.DENSE_DIM = 768
        self.HIDDEN_DIM_DROPOUT_PROB = 0.3
        self.NUMBER_OF_LABELS = 2

        self.code_bert = RobertaModel.from_pretrained("microsoft/codebert-base", num_labels=2)

        self.linear = nn.Linear(self.HIDDEN_DIM, self.DENSE_DIM)
        self.relu = nn.ReLU()
        self.drop_out = nn.Dropout(self.HIDDEN_DIM_DROPOUT_PROB)
        self.out_proj = nn.Linear(self.DENSE_DIM, self.NUMBER_OF_LABELS)

    def forward(self, input_batch, mask_batch):
        embeddings = self.code_bert(input_ids=input_batch, attention_mask=mask_batch).last_hidden_state[:, 0, :]

        x = embeddings
        x = self.drop_out(x)
        x = self.linear(x)
        x = self.relu(x)
        x = self.drop_out(x)
        x = self.out_proj(x)

        return x


class VariantSevenClassifier(nn.Module):
    """An 1D Convulational Neural Network for Sentence Classification."""

    def __init__(self,
                 embed_dim=768,
                 filter_sizes=[2, 3, 4],
                 num_filters=[100, 100, 100],
                 num_classes=2,
                 dropout=0.5):
        super(VariantSevenClassifier, self).__init__()
        self.embed_dim = embed_dim
        print(filter_sizes)
        # Conv Network
        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(in_channels=self.embed_dim,
                      out_channels=num_filters[i],
                      kernel_size=filter_sizes[i])
            for i in range(len(filter_sizes))
        ])
        # Fully-connected layer and Dropout
        self.fc = nn.Linear(2 * np.sum(num_filters), num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, before_batch, after_batch, need_final_feature=False):
        """Perform a forward pass through the network.

        Args:
            input_ids (torch.Tensor): A tensor of token ids with shape
                (batch_size, max_sent_length)

        Returns:
            logits (torch.Tensor): Output logits with shape (batch_size,
                n_classes)
        """

        # Get embeddings from `input_ids`. Output shape: (b, max_len, embed_dim)
        x_embed_before = before_batch
        # batch, file, hidden_dim

        # Permute `x_embed` to match input shape requirement of `nn.Conv1d`.
        # Output shape: (b, embed_dim, max_len)
        x_reshaped_before = x_embed_before.permute(0, 2, 1)

        # Apply CNN and ReLU. Output shape: (b, num_filters[i], L_out)
        x_conv_list_before = [F.relu(conv1d(x_reshaped_before)) for conv1d in self.conv1d_list]

        # Max pooling. Output shape: (b, num_filters[i], 1)
        x_pool_list_before = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
                              for x_conv in x_conv_list_before]

        # Concatenate x_pool_list to feed the fully connected layer.
        # Output shape: (b, sum(num_filters))
        x_fc_before = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list_before],
                                dim=1)

        # # Compute logits. Output shape: (b, n_classes)
        # out = self.fc(self.dropout(x_fc_before))

        ############################################

        x_embed_after = after_batch

        # Permute `x_embed` to match input shape requirement of `nn.Conv1d`.
        # Output shape: (b, embed_dim, max_len)
        x_reshaped_after = x_embed_after.permute(0, 2, 1)

        # Apply CNN and ReLU. Output shape: (b, num_filters[i], L_out)
        x_conv_list_after = [F.relu(conv1d(x_reshaped_after)) for conv1d in self.conv1d_list]

        # Max pooling. Output shape: (b, num_filters[i], 1)
        x_pool_list_after = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
                             for x_conv in x_conv_list_after]

        # Concatenate x_pool_list to feed the fully connected layer.
        # Output shape: (b, sum(num_filters))
        x_fc_after = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list_after],
                               dim=1)

        # Compute logits. Output shape: (b, n_classes)

        x_fc = torch.cat([x_fc_before, x_fc_after], axis=1)
        final_feature = x_fc

        out = self.fc(self.dropout(x_fc))

        if need_final_feature:
            return out, final_feature
        else:
            return out


class VariantOneClassifier(nn.Module):
    def __init__(self):
        super(VariantOneClassifier, self).__init__()
        self.HIDDEN_DIM = 768  # 隐藏层维度768维
        self.DENSE_DIM = 768  # 全连接层维度768维
        self.HIDDEN_DIM_DROPOUT_PROB = 0.3  # 隐藏层之间添加的Dropout层的丢弃概率
        self.NUMBER_OF_LABELS = 2  # 输出层的神经元数量，表示分类任务的类别数

        self.embedding = nn.Linear(self.HIDDEN_DIM, self.DENSE_DIM)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=self.DENSE_DIM, nhead=8)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=2)
        self.relu = nn.ReLU()

        self.drop_out = nn.Dropout(self.HIDDEN_DIM_DROPOUT_PROB)
        self.out_proj = nn.Linear(self.DENSE_DIM, self.NUMBER_OF_LABELS)

    def forward(self, embedding_batch, need_final_feature=False):
        x = embedding_batch
        x = self.drop_out(x)
        x = self.embedding(x)
        x = x.unsqueeze(1)  # 添加一个维度用于 Transformer 的 batch_first 处理
        x = self.transformer(x)
        x = x.squeeze(1)  # 移除多余的维度
        x = self.relu(x)
        final_feature = x
        x = self.drop_out(x)
        x = self.out_proj(x)

        if need_final_feature:
            return x, final_feature
        else:
            return x


class VariantOneFinetuneClassifier(nn.Module):
    def __init__(self):
        super(VariantOneFinetuneClassifier, self).__init__()

        self.code_bert = RobertaModel.from_pretrained("microsoft/codebert-base", num_labels=2)
        self.classifier = VariantOneClassifier()

    def forward(self, input_batch, mask_batch):
        embeddings = self.code_bert(input_ids=input_batch, attention_mask=mask_batch)
        embeddings = embeddings.last_hidden_state[:, 0, :]
        out = self.classifier(embeddings)
        return out

    def freeze_codebert(self):
        if not isinstance(self, nn.DataParallel):
            for param in self.code_bert.parameters():
                param.requires_grad = False  # 暂停梯度的更新
        else:
            for param in self.module.code_bert.parameters():
                param.requires_grad = False


class VariantFiveClassifier(nn.Module):
    def __init__(self):
        super(VariantFiveClassifier, self).__init__()
        self.HIDDEN_DIM = 768
        self.DENSE_DIM = 128
        self.HIDDEN_DIM_DROPOUT_PROB = 0.5
        self.NUMBER_OF_LABELS = 2
        self.linear = nn.Linear(2 * self.HIDDEN_DIM, self.DENSE_DIM)
        self.relu = nn.ReLU()

        self.drop_out = nn.Dropout(self.HIDDEN_DIM_DROPOUT_PROB)
        self.out_proj = nn.Linear(self.DENSE_DIM, self.NUMBER_OF_LABELS)

    def forward(self, before_batch, after_batch, need_final_feature=False):
        combined = torch.cat([before_batch, after_batch], dim=1)
        x = combined
        x = self.drop_out(x)
        x = self.linear(x)
        x = self.relu(x)
        final_feature = x
        x = self.drop_out(x)
        x = self.out_proj(x)

        if need_final_feature:
            return x, final_feature
        else:
            return x


class VariantFiveFineTuneClassifier(nn.Module):
    def __init__(self):
        super(VariantFiveFineTuneClassifier, self).__init__()
        self.code_bert = RobertaModel.from_pretrained("microsoft/codebert-base", num_labels=2)
        self.classifier = VariantFiveClassifier()

    def forward(self, added_input, added_mask, removed_input, removed_mask):
        added_embeddings = self.code_bert(input_ids=added_input, attention_mask=added_mask).last_hidden_state[:, 0, :]
        removed_embeddings = self.code_bert(input_ids=removed_input, attention_mask=removed_mask).last_hidden_state[:,
                             0, :]
        out = self.classifier(added_embeddings, removed_embeddings)
        return out

    def freeze_codebert(self):
        if not isinstance(self, nn.DataParallel):
            for param in self.code_bert.parameters():
                param.requires_grad = False
        else:
            for param in self.module.code_bert.parameters():
                param.requires_grad = False


class VariantEightClassifier(nn.Module):
    def __init__(self):
        super(VariantEightClassifier, self).__init__()
        self.input_size = 768
        self.hidden_size = 128
        self.HIDDEN_DIM_DROPOUT_PROB = 0.3
        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            batch_first=True,
                            bidirectional=True)
        self.linear = nn.Linear(4 * self.hidden_size, self.hidden_size)

        self.relu = nn.ReLU()

        self.drop_out = nn.Dropout(self.HIDDEN_DIM_DROPOUT_PROB)

        self.out_proj = nn.Linear(self.hidden_size, 2)

    def forward(self, before_batch, after_batch, need_final_feature=False):
        # self.lstm.flatten_parameters()
        before_out, (before_final_hidden_state, _) = self.lstm(before_batch)
        before_vector = before_out[:, 0]

        after_out, (after_final_hidden_state, _) = self.lstm(after_batch)
        after_vector = after_out[:, 0]

        x = self.linear(torch.cat([before_vector, after_vector], axis=1))

        x = self.relu(x)
        final_feature = x

        x = self.drop_out(x)

        out = self.out_proj(x)

        if need_final_feature:
            return out, final_feature
        else:
            return out


class VariantEightLstmClassifier(nn.Module):
    def __init__(self):
        super(VariantEightLstmClassifier, self).__init__()
        self.input_size = 768
        self.hidden_size = 128
        self.HIDDEN_DIM_DROPOUT_PROB = 0.3
        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            batch_first=True,
                            bidirectional=False)
        self.linear = nn.Linear(2 * self.hidden_size, self.hidden_size)

        self.relu = nn.ReLU()

        self.drop_out = nn.Dropout(self.HIDDEN_DIM_DROPOUT_PROB)

        self.out_proj = nn.Linear(self.hidden_size, 2)

    def forward(self, before_batch, after_batch, need_final_feature=False):
        # self.lstm.flatten_parameters()
        before_out, (before_final_hidden_state, _) = self.lstm(before_batch)
        before_vector = before_out[:, 0]

        after_out, (after_final_hidden_state, _) = self.lstm(after_batch)
        after_vector = after_out[:, 0]

        x = self.linear(torch.cat([before_vector, after_vector], axis=1))

        x = self.relu(x)
        final_feature = x

        x = self.drop_out(x)

        out = self.out_proj(x)

        if need_final_feature:
            return out, final_feature
        else:
            return out


class VariantEightGruClassifier(nn.Module):
    def __init__(self):
        super(VariantEightGruClassifier, self).__init__()
        self.input_size = 768
        self.hidden_size = 128
        self.HIDDEN_DIM_DROPOUT_PROB = 0.3
        self.gru = nn.GRU(input_size=self.input_size,
                          hidden_size=self.hidden_size,
                          batch_first=True,
                          bidirectional=False)
        self.linear = nn.Linear(2 * self.hidden_size, self.hidden_size)

        self.relu = nn.ReLU()

        self.drop_out = nn.Dropout(self.HIDDEN_DIM_DROPOUT_PROB)

        self.out_proj = nn.Linear(self.hidden_size, 2)

    def forward(self, before_batch, after_batch, need_final_feature=False):
        # self.lstm.flatten_parameters()
        before_out, before_final_hidden_state = self.gru(before_batch)
        before_vector = before_out[:, 0]

        after_out, after_final_hidden_state = self.gru(after_batch)
        after_vector = after_out[:, 0]

        x = self.linear(torch.cat([before_vector, after_vector], axis=1))

        x = self.relu(x)
        final_feature = x

        x = self.drop_out(x)

        out = self.out_proj(x)

        if need_final_feature:
            return out, final_feature
        else:
            return out


class VariantSixFineTuneClassifier(nn.Module):
    def __init__(self):
        super(VariantSixFineTuneClassifier, self).__init__()
        self.HIDDEN_DIM = 768
        self.code_bert = RobertaModel.from_pretrained("microsoft/codebert-base", num_labels=2)
        self.classifier = VariantSixClassifier()

    def forward(self, added_input_list_batch, added_mask_list_batch, removed_input_list_batch, removed_mask_list_batch):
        d1, d2, d3 = added_input_list_batch.shape
        added_input_list_batch = torch.reshape(added_input_list_batch, (d1 * d2, d3))
        added_mask_list_batch = torch.reshape(added_mask_list_batch, (d1 * d2, d3))
        added_embeddings = self.code_bert(input_ids=added_input_list_batch,
                                          attention_mask=added_mask_list_batch).last_hidden_state[:, 0, :]
        added_embeddings = torch.reshape(added_embeddings, (d1, d2, self.HIDDEN_DIM))

        removed_input_list_batch = torch.reshape(removed_input_list_batch, (d1 * d2, d3))
        removed_mask_list_batch = torch.reshape(removed_mask_list_batch, (d1 * d2, d3))
        removed_embeddings = self.code_bert(input_ids=removed_input_list_batch,
                                            attention_mask=removed_mask_list_batch).last_hidden_state[:, 0, :]
        removed_embeddings = torch.reshape(removed_embeddings, (d1, d2, self.HIDDEN_DIM))

        out = self.classifier(added_embeddings, removed_embeddings)

        return out

    def freeze_codebert(self):
        if not isinstance(self, nn.DataParallel):
            for param in self.code_bert.parameters():
                param.requires_grad = False
        else:
            for param in self.module.code_bert.parameters():
                param.requires_grad = False


class EncoderRNN(nn.Module):
    def __init__(self,
                 emb_dim,
                 h_dim,
                 batch_first=True):
        super(EncoderRNN, self).__init__()
        self.h_dim = h_dim
        # self.embed = nn.Embedding(v_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim,
                            h_dim,
                            batch_first=batch_first,
                            bidirectional=True)

    def init_hidden(self, b_size):
        h0 = Variable(torch.zeros(1 * 2, b_size, self.h_dim))
        c0 = Variable(torch.zeros(1 * 2, b_size, self.h_dim))
        h0 = h0.cuda()
        c0 = c0.cuda()
        return h0, c0

    def forward(self, sentence):
        # self.lstm.flatten_parameters()
        hidden = self.init_hidden(sentence.size(0))
        emb = sentence
        out, hidden = self.lstm(emb, hidden)
        out = out[:, :, :self.h_dim] + out[:, :, self.h_dim:]
        return out


class Attn(nn.Module):
    def __init__(self, h_dim):
        super(Attn, self).__init__()
        self.h_dim = h_dim
        self.main = nn.Sequential(nn.Linear(h_dim, 24), nn.ReLU(True),
                                  nn.Linear(24, 1))

    def forward(self, encoder_outputs):
        b_size = encoder_outputs.size(0)
        attn_ene = self.main(encoder_outputs.reshape(-1, self.h_dim))  # (b, s, h) -> (b * s, 1)
        return F.softmax(attn_ene.view(b_size, -1),
                         dim=1).unsqueeze(2)  # (b*s, 1) -> (b, s, 1)


class AttnClassifier(nn.Module):
    def __init__(self, h_dim, c_num):
        super(AttnClassifier, self).__init__()
        self.attn1 = Attn(h_dim)
        self.attn2 = Attn(h_dim)
        self.linear = nn.Linear(2 * h_dim, h_dim)
        self.output = nn.Linear(h_dim, c_num)

    def forward(self, a_output, b_output):
        a_attn = self.attn1(a_output)  # (b, s, 1)
        b_attn = self.attn2(b_output)  # ()  #(b, s, 1)
        a_feats = (a_output * a_attn).sum(dim=1)  # (b, s, h) -> (b, h)
        b_feats = (b_output * b_attn).sum(dim=1)
        feats = torch.cat((a_feats, b_feats), 1)
        o_feats = self.linear(feats)
        out = self.output(o_feats)
        return F.log_softmax(out, -1), a_attn, b_attn


class VariantEightAttentionClassifier(nn.Module):
    def __init__(self):
        super(VariantEightAttentionClassifier, self).__init__()
        self.EMBEDDING_DIM = 768
        self.HIDDEN_DIM = 128
        self.NUMBER_OF_LABELS = 2
        self.before_encoder = EncoderRNN(self.EMBEDDING_DIM, self.HIDDEN_DIM)
        self.after_encoder = EncoderRNN(self.EMBEDDING_DIM, self.HIDDEN_DIM)

        self.attn1 = Attn(self.HIDDEN_DIM)
        self.attn2 = Attn(self.HIDDEN_DIM)
        self.linear = nn.Linear(2 * self.HIDDEN_DIM, self.HIDDEN_DIM)
        self.output = nn.Linear(self.HIDDEN_DIM, self.NUMBER_OF_LABELS)

    def forward(self, before_batch, after_batch):
        before_out = self.before_encoder(before_batch)
        after_out = self.after_encoder(after_batch)
        a_attn = self.attn1(after_out)  # (b, s, 1)
        b_attn = self.attn2(before_out)  # ()  #(b, s, 1)
        a_feats = (after_out * a_attn).sum(dim=1)  # (b, s, h) -> (b, h)
        b_feats = (before_out * b_attn).sum(dim=1)
        feats = torch.cat((a_feats, b_feats), 1)
        o_feats = self.linear(feats)
        out = self.output(o_feats)
        return out


class VariantEightFineTuneOnlyClassifier(nn.Module):
    def __init__(self):
        super(VariantEightFineTuneOnlyClassifier, self).__init__()
        self.HIDDEN_DIM = 768
        self.DENSE_DIM = 768
        self.HIDDEN_DIM_DROPOUT_PROB = 0.3
        self.NUMBER_OF_LABELS = 2

        self.code_bert = RobertaModel.from_pretrained("microsoft/codebert-base", num_labels=2)

        self.linear = nn.Linear(self.HIDDEN_DIM, self.DENSE_DIM)
        self.relu = nn.ReLU()
        self.drop_out = nn.Dropout(self.HIDDEN_DIM_DROPOUT_PROB)
        self.out_proj = nn.Linear(self.DENSE_DIM, self.NUMBER_OF_LABELS)

    def forward(self, input_batch, mask_batch):
        embeddings = self.code_bert(input_ids=input_batch, attention_mask=mask_batch).last_hidden_state[:, 0, :]

        x = embeddings
        x = self.drop_out(x)
        x = self.linear(x)
        x = self.relu(x)
        x = self.drop_out(x)
        x = self.out_proj(x)

        return x


class VariantSeventFineTuneOnlyClassifier(nn.Module):
    def __init__(self):
        super(VariantSeventFineTuneOnlyClassifier, self).__init__()
        self.HIDDEN_DIM = 768
        self.DENSE_DIM = 768
        self.HIDDEN_DIM_DROPOUT_PROB = 0.3
        self.NUMBER_OF_LABELS = 2

        self.code_bert = RobertaModel.from_pretrained("microsoft/codebert-base", num_labels=2)

        self.linear = nn.Linear(self.HIDDEN_DIM, self.DENSE_DIM)
        self.relu = nn.ReLU()
        self.drop_out = nn.Dropout(self.HIDDEN_DIM_DROPOUT_PROB)
        self.out_proj = nn.Linear(self.DENSE_DIM, self.NUMBER_OF_LABELS)

    def forward(self, input_batch, mask_batch, need_final_feature=False):
        embeddings = self.code_bert(input_ids=input_batch, attention_mask=mask_batch).last_hidden_state[:, 0, :]

        x = embeddings
        x = self.drop_out(x)
        x = self.linear(x)
        x = self.relu(x)
        final_feature = x
        x = self.drop_out(x)
        x = self.out_proj(x)

        if need_final_feature:
            return x, final_feature
        else:
            return x


class ScaledDotProductAttention(nn.Module):
    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, query, key, value, mask=None):
        d_k = query.size(-1)  # 获取query的维度
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # 计算注意力分数

        if mask is not None:
            scores = scores.masked.fill(mask == 0, -1e9)

        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        output = torch.matmul(attention, value)
        return output, attention


# class MultiHeadAttention(nn.Module):
#     def __init__(self, num_heads, d_model, dropout=0.1):
#         super(MultiHeadAttention, self).__init__()
#         assert d_model % num_heads == 0  # 确保d_model可以被num_heads整除
#
#         self.d_k = d_model // num_heads
#         self.num_heads = num_heads
#         self.query_linear = nn.Linear(d_model, d_model)
#         self.key_linear = nn.Linear(d_model, d_model)
#         self.value_linear = nn.Linear(d_model, d_model)
#         self.dropout = nn.Dropout(dropout)
#         self.out = nn.Linear(d_model, d_model)
#         self.attention = ScaledDotProductAttention()
#
#     def forward(self, query, key, value, mask=None):
#         if mask is not None:
#             # 同样的mask应用于所有头
#             mask = mask.unsqueeze(1)
#
#         batch_size = query.size(0)
#
#         # 1) 执行所有线性变换
#         query = self.query_linear(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
#         key = self.key_linear(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
#         value = self.value_linear(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
#
#         # 2) 应用注意力机制
#         x, attn = self.attention(query, key, value, mask=mask)
#
#         # 3) “Concat”使用view拼接和应用最终的线性层
#         x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
#         return self.out(x)

"""把feature_2变为feature_1"""
# class MatchAttention(nn.Module):
#     def __init__(self, feature_dim_1, feature_dim_2):
#         super(MatchAttention, self).__init__()
#         # 为第二组特征设置一个线性层以将其维度与第一组特征匹配
#         self.linear = nn.Linear(feature_dim_2, feature_dim_1)
#
#     def forward(self, features_1, features_2):
#         # 调整 features_2 的维度以匹配 features_1
#         transformed_features_2 = self.linear(features_2)
#
#         # 计算相似度矩阵
#         similarity_matrix = torch.matmul(features_1, transformed_features_2.transpose(-2, -1))
#
#         # 计算注意力权重
#         attention_weights_1 = F.softmax(similarity_matrix, dim=-1)
#         attention_weights_2 = F.softmax(similarity_matrix.transpose(-2, -1), dim=-1)
#
#         # 加权特征
#         attended_features_1 = torch.matmul(attention_weights_1, transformed_features_2)
#         attended_features_2 = torch.matmul(attention_weights_2, features_1)
#
#         return attended_features_1, attended_features_2

""""把第一个维度映射为第二个维度"""


class MatchAttention(nn.Module):
    def __init__(self, feature_dim_1, feature_dim_2):
        super(MatchAttention, self).__init__()
        self.linear = nn.Linear(feature_dim_1, feature_dim_2)
        self.linear2 = nn.Linear(feature_dim_2, feature_dim_1)  # 把feature_1线性变换成原来的形状

    def forward(self, features_1, features_2):
        # 调整 features_2 的维度以匹配 features_1
        transformed_features_1 = self.linear(features_1)

        # 计算相似度矩阵
        similarity_matrix = torch.matmul(transformed_features_1, features_2.transpose(-2, -1))

        # 计算注意力权重
        attention_weights_1 = F.softmax(similarity_matrix, dim=-1)
        attention_weights_2 = F.softmax(similarity_matrix, dim=-1)

        # 加权特征
        attended_features_1 = torch.matmul(attention_weights_1, features_2)
        attended_features_2 = torch.matmul(attention_weights_2, transformed_features_1)

        attended_features_2 = self.linear2(attended_features_2)
        return attended_features_1, attended_features_2


"""GMN的match attention"""
# class FeatureMapping(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(FeatureMapping, self).__init__()
#         self.fc = nn.Linear(input_dim, output_dim)
#
#     def forward(self, x):
#         return self.fc(x)
#
# class MatchAttention(nn.Module):
#     def __init__(self, dim1, dim2, common_dim):
#         super(MatchAttention, self).__init__()
#         # 映射到共同维度
#         self.map1 = FeatureMapping(dim1, common_dim)
#         self.map2 = FeatureMapping(dim2, common_dim)
#
#     def forward(self, feature_1, feature_2):
#         # 映射特征
#         mapped_feature_1 = self.map1(feature_1)
#         mapped_feature_2 = self.map2(feature_2)
#
#         # 计算匹配分数
#         match_score = torch.bmm(mapped_feature_1, mapped_feature_2.transpose(1, 2))
#
#         # 应用softmax获取注意力权重
#         attention_weights = F.softmax(match_score, dim=2)
#
#         # 使用注意力权重加权特征
#         weighted_feature = torch.bmm(attention_weights, mapped_feature_2)
#
#         return weighted_feature

"""mutilhead-attention修改的match attention"""


# class Attention(nn.Module):
#     def __init__(self, nb_head, size_per_head):
#         super(Attention, self).__init__()
#         self.nb_head = nb_head
#         self.size_per_head = size_per_head
#         self.device = 'cuda:0'  # 添加一个属性以存储设备信息
#
#     def forward(self, Q, K, V, Q_len=None, V_len=None):
#         # 将输入张量移动到设备上
#         Q = Q.to(self.device)
#         K = K.to(self.device)
#         V = V.to(self.device)
#
#         # 使用 self.dense 方法，确保输出位于相同的设备上
#         Q = self.dense(Q, self.nb_head * self.size_per_head, bias=False)
#         K = self.dense(K, self.nb_head * self.size_per_head, bias=False)
#         V = self.dense(V, self.nb_head * self.size_per_head, bias=False)
#
#         # 转换形状并进行注意力计算
#         Q = Q.view(-1, Q.size(1), self.nb_head, self.size_per_head).transpose(1, 2)
#         K = K.view(-1, K.size(1), self.nb_head, self.size_per_head).transpose(1, 2)
#         V = V.view(-1, V.size(1), self.nb_head, self.size_per_head).transpose(1, 2)
#
#         A = torch.matmul(Q, K.transpose(-2, -1)) / (self.size_per_head ** 0.5)
#         A = F.softmax(A, dim=-1)
#
#         O = torch.matmul(A, V).transpose(1, 2).contiguous()
#         O = O.view(-1, O.size(1), self.nb_head * self.size_per_head)
#
#         # 如果提供了长度信息，则进行 mask 操作
#         if Q_len is not None:
#             O = self.mask(O, Q_len, mode='mul')
#
#         return O
#
#     def dense(self, inputs, output_size, bias=True):
#         linear_layer = nn.Linear(inputs.size(-1), output_size, bias=bias).to(self.device)
#         return linear_layer(inputs)
#
#     def mask(self, inputs, seq_len, mode='mul'):
#         mask = torch.BoolTensor(seq_len[:, None] > torch.arange(inputs.size(-2))[None, :]).to(inputs.device)
#         mask = mask.unsqueeze(-1).expand_as(inputs)
#         mask = mask.type(inputs.type())
#         if mode == 'mul':
#             return inputs * mask
#         elif mode == 'add':
#             return inputs - (1 - mask) * 1e12

class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, num_layers=2):
        super(CNNFeatureExtractor, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(
                nn.Conv1d(input_dim if i == 0 else output_dim, output_dim, kernel_size=kernel_size, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(kernel_size=2))
        self.cnn = nn.Sequential(*layers)

    def forward(self, x):
        return self.cnn(x)

class EnsembleModel(nn.Module):  # 这个模型主要是集成不同的特征并且进行分类
    def __init__(self, ablation_study=False, variant_to_drop=None, num_layers=2, nhead=4):
        super(EnsembleModel, self).__init__()
        self.FEATURE_DIM = 768  # 特征的维度设置为768
        self.DENSE_DIM = 128  # 密集层的维度设置为128
        self.CNN_FEATURE_DIM = 300  # CNN特征维度设置为300
        self.HIDDEN_DIM_DROPOUT_PROB = 0.5  # 隐藏层dropout设置为0.3
        self.NUMBER_OF_LABELS = 2  # 输出标签的数量设置为2，就是一个二分类问题
        self.your_size_per_head = 128  # your_size_per_head


        """Transformer modules"""
        self.transformer_encoder_1 = TransformerEncoder(
            TransformerEncoderLayer(d_model=3 * self.FEATURE_DIM, nhead=nhead),
            num_layers=num_layers
        )
        self.transformer_encoder_2 = TransformerEncoder(
            TransformerEncoderLayer(d_model=4 * self.FEATURE_DIM, nhead=nhead),
            num_layers=num_layers
        )

        """CNN modules"""
        # self.cnn_1 = CNNFeatureExtractor(input_dim=3 * self.FEATURE_DIM, output_dim=self.FEATURE_DIM)
        # self.cnn_2 = CNNFeatureExtractor(input_dim=4 * self.FEATURE_DIM, output_dim=self.FEATURE_DIM)
        self.transformer_1 = Transformer(d_model=3*self.FEATURE_DIM,nhead=nhead,num_encoder_layers=num_layers,num_decoder_layers=num_layers)
        self.transformer_2 = Transformer(d_model=4*self.FEATURE_DIM,nhead=nhead,num_encoder_layers=num_layers,num_decoder_layers=num_layers)
        """GMN的match attention"""
        # self.match_attention = MatchAttention(dim1=3*self.FEATURE_DIM, dim2=4*self.FEATURE_DIM, common_dim=4*self.FEATURE_DIM)

        """多头match"""
        # self.attention_layer = Attention(nb_head=nhead, size_per_head=self.your_size_per_head)
        # self.attention_layer = self.attention_layer.to(self.device)
        """自己的match attention"""
        self.match_attention = MatchAttention(feature_dim_1=3 * self.FEATURE_DIM, feature_dim_2=4 * self.FEATURE_DIM)
        # """对于feature_extractor进行多头的transformer"""
        # self.transformer_encoder_3 = TransformerEncoder(
        #     TransformerEncoderLayer(d_model=4*self.FEATURE_DIM,nhead=nhead),
        #     num_layers=num_layers
        # )
        # self.transformer_encoder_4 = TransformerEncoder(
        #     TransformerEncoderLayer(d_model=2*self.FEATURE_DIM,nhead=nhead),
        #     num_layers=num_layers
        # )

        # need 2 linear layer to project CNN feature dim to 768
        # 1 for variant 3
        # 1 for variant 7
        self.l1 = nn.Linear(self.CNN_FEATURE_DIM, self.FEATURE_DIM)  # l1和l2:将CNN特征维度转换为768维，输入的维度为CNN_FEATURE为300维
        self.l2 = nn.Linear(self.CNN_FEATURE_DIM * 2, self.FEATURE_DIM)  # 输入为CNNC_FEATURE_DIM * 2为600维

        # need 1 linear layer to project variant 5 feature to 768

        self.l3 = nn.Linear(self.DENSE_DIM, self.FEATURE_DIM)  # l3,l4将DENSE特征维度转换为768维

        # need 1 linear layer to project variant 8 feature to 768
        self.l4 = nn.Linear(self.DENSE_DIM, self.FEATURE_DIM)

        # 1 layer to combine
        self.ablation_study = ablation_study  # 将传入模型ablation_study参数的值保存到模型内部

        """最终七个特征"""
        if not self.ablation_study:
            self.l5 = nn.Linear(7 * self.FEATURE_DIM, self.FEATURE_DIM) #l5联合所有特征的线性层
        else:
            self.l5 = nn.Linear((7 - len(variant_to_drop)) * self.FEATURE_DIM, self.FEATURE_DIM)

        """多头的match attention"""
        # if not self.ablation_study:
        #     self.l5 = nn.Linear(512, self.FEATURE_DIM)  # l5联合所有特征的线性层
        # else:
        #     self.l5 = nn.Linear(512, self.FEATURE_DIM)

        """最终六个特征"""
        # if not self.ablation_study:
        #     self.l5 = nn.Linear(6 * self.FEATURE_DIM, self.FEATURE_DIM) #l5联合所有特征的线性层
        # else:
        #     self.l5 = nn.Linear((6 - len(variant_to_drop)) * self.FEATURE_DIM, self.FEATURE_DIM)

        """最终两个特征的话"""
        # if not self.ablation_study:
        #     self.l5 = nn.Linear(2 * self.FEATURE_DIM, self.FEATURE_DIM) #l5联合所有特征的线性层
        # else:
        #     self.l5 = nn.Linear((2 - len(variant_to_drop)) * self.FEATURE_DIM, self.FEATURE_DIM)

        """最终三个特征的话"""
        # if not self.ablation_study:
        #     self.l5 = nn.Linear(3 * self.FEATURE_DIM, self.FEATURE_DIM)  # l5联合所有特征的线性层
        # else:
        #     self.l5 = nn.Linear((3 - len(variant_to_drop)) * self.FEATURE_DIM, self.FEATURE_DIM)

        """最终四个特征的话"""
        # if not self.ablation_study:
        #     self.l5 = nn.Linear(4 * self.FEATURE_DIM, self.FEATURE_DIM)  # l5联合所有特征的线性层
        # else:
        #     self.l5 = nn.Linear((4 - len(variant_to_drop)) * self.FEATURE_DIM, self.FEATURE_DIM)
        #

        """最终八个特征的话"""
        # if not self.ablation_study:
        #     self.l5 = nn.Linear(8 * self.FEATURE_DIM, self.FEATURE_DIM)  # l5联合所有特征的线性层
        # else:
        #     self.l5 = nn.Linear((8 - len(variant_to_drop)) * self.FEATURE_DIM, self.FEATURE_DIM)

        self.variant_to_drop = variant_to_drop
        self.relu = nn.ReLU()  # ReLU激活函数

        self.drop_out = nn.Dropout(self.HIDDEN_DIM_DROPOUT_PROB)  # dropout层
        self.out_proj = nn.Linear(self.FEATURE_DIM, self.NUMBER_OF_LABELS)
        # 定义了一个全连接层，将模型的中间特征从self.FEATURE_DIM转换到self.NUMBER_OF_LABELS
        # self.num_heads = num_heads  #添加头的数量属性
        # self.attention = nn.MultiheadAttention(embed_dim=768,num_heads=8)
        # self.attention_3 = nn.MultiheadAttention(embed_dim=300,num_heads=4)
        # self.attention_5 = nn.MultiheadAttention(embed_dim=128, num_heads=4)
        # self.attention_7 = nn.MultiheadAttention(embed_dim=600, num_heads=8)
        # self.attention_8 = nn.MultiheadAttention(embed_dim=128, num_heads=4)

    def forward(self, feature_1, feature_2, feature_3, feature_5, feature_6, feature_7, feature_8, need_features=False):

        # 这个forward将一个集合特征作为输入并最终输出一个向量，向量的维度由NUMBER_OF_LABELS决定

        # print("feature_1的提取完的形状", feature_1.shape)
        # print("feature_2的提取完的形状", feature_2.shape)
        # print("feature_3的提取完的形状", feature_3.shape)
        # print("feature_5的提取完的形状", feature_5.shape)
        # print("feature_6的提取完的形状", feature_6.shape)
        # print("feature_7的提取完的形状", feature_7.shape)
        # print("feature_8的提取完的形状", feature_8.shape)

        feature_3 = self.l1(feature_3)  # 使用线性层l1对feature3进行变换
        feature_7 = self.l2(feature_7)  # 使用线性层l2对feature7进行变换
        feature_5 = self.l3(feature_5)  # 使用线性层l3对feature5进行变换
        feature_8 = self.l4(feature_8)  # 使用线性层l4对feature8进行变换

        feature_1 = feature_1.unsqueeze(1)  # shape: (64, 1, 768)
        feature_2 = feature_2.unsqueeze(1)  # shape: (64, 1, 768)
        feature_3 = feature_3.unsqueeze(1)  # shape: (64, 1, 768)
        feature_5 = feature_5.unsqueeze(1)  # shape: (64, 1, 768)
        feature_6 = feature_6.unsqueeze(1)  # shape: (64, 1, 768)
        feature_7 = feature_7.unsqueeze(1)  # shape: (64, 1, 768)
        feature_8 = feature_8.unsqueeze(1)

        """根据context-depend进行多头的transformer"""
        feature_list_1 = [feature_1, feature_2, feature_3]
        feature_list_2 = [feature_5, feature_6, feature_7, feature_8]
        feature_set_1 = torch.cat(feature_list_1, axis=2)
        feature_set_2 = torch.cat(feature_list_2, axis=2)

        """transformer model"""
        transformed_features_1 = self.transformer_encoder_1(feature_set_1)
        transformed_features_2 = self.transformer_encoder_2(feature_set_2)

        """cnn moduel"""
        # transformed_features_1 = self.cnn_1(feature_set_1)
        # transformed_features_2 = self.cnn_2(feature_set_2)

        """先match attention再transformer"""
        # attended_features_1, attended_features_2 = self.match_attention(feature_set_1, feature_set_2)
        # #Apply transformer heads to each feature set
        # transformed_features_1 = self.transformer_encoder_1(attended_features_2)
        # transformed_features_2 = self.transformer_encoder_2(attended_features_1)

        # 自己写的match attention 模块
        attended_features_1, attended_features_2 = self.match_attention(transformed_features_1, transformed_features_2)
        combined_features = torch.cat([attended_features_1, attended_features_2], axis=2)
        # outputs = self.attention_layer(transformed_features_1, transformed_features_2, transformed_features_2)

        # print("多头match_attention的outputs的shape: ", outputs.shape)
        # combined_features = torch.cat([transformed_features_1, transformed_features_2], axis=2)

        combined_features = combined_features.squeeze(1)
        combined_features = self.drop_out(combined_features)
        combined_features = self.l5(combined_features)

        # Pass through the classifier
        logits = self.relu(combined_features)

        logits = self.drop_out(logits)
        logits = self.out_proj(logits)

        # GMN的match attention 模块
        # weighted_feature = self.match_attention(transformed_features_1, transformed_features_2)
        # weighted_feature = weighted_feature.squeeze(1)
        # weighted_feature = self.drop_out(weighted_feature)
        # weighted_feature = self.l5(weighted_feature)
        #
        # #Pass through the classifier
        # logits = self.relu(weighted_feature)
        # logits = self.drop_out(logits)
        # logits = self.out_proj(logits)

        # print("加权输出的特征：",weighted_feature.shape)  # 输出加权特征的形状

        """"根据提取方法进行一个多头的transformer"""
        # feature_list_1 = [feature_1,feature_2,feature_5,feature_6]
        # feature_list_2 = [feature_3,feature_7]
        # feature_set_1 = torch.cat(feature_list_1,axis=2)
        # feature_set_2 =torch.cat(feature_list_2,axis=2)
        #
        # #调用tranformer的API
        # transformed_features_1 = self.transformer_encoder_3(feature_set_1)
        # transformed_features_2 = self.transformer_encoder_4(feature_set_2)
        #
        # #将三种不同提取特征的方法进行一个组合
        # combined_features = torch.cat([transformed_features_1,transformed_features_2,feature_8],axis=2)
        #
        # #应用drop out和线性变化
        # combined_features = combined_features.squeeze(1)
        # combined_features = self.drop_out(combined_features)
        # combined_features = self.l5(combined_features)
        #
        # #通过分类器进行分类
        # logits = self.relu(combined_features)
        # logits = self.drop_out(logits)
        # logits = self.out_proj(logits)

        # Return features if needed for PCA analysis or similar
        if not need_features:  # 如果不需要特征
            return logits  # 直接输出变化后的结果
        else:
            return logits, combined_features  # 把输出后的结果和pca_features一起输出
        # print("feature_3的线性变换完的形状", feature_5.shape)
        # print("feature_7的线性变换完的形状", feature_6.shape)
        # print("feature_5的线性变换完的形状", feature_7.shape)
        # print("feature_8的线性变换完的形状", feature_8.shape)

        """对每个特征都进行自注意力"""
        # feature_1 = feature_1.unsqueeze(0)
        # # 把feature_1作为query,feature_2和feature3作为key和value
        # attn_output1, _ = self.attention(feature_1, feature_1, feature_1)
        # attn_output1 = attn_output1.squeeze(0)
        # # #现在可以把attn_output作为新的feature_1
        # feature_1 = attn_output1
        #
        # feature_2 = feature_2.unsqueeze(0)
        # # 把feature_1作为query,feature_2和feature3作为key和value
        # attn_output2, _ = self.attention(feature_2, feature_2, feature_2)
        # attn_output2 = attn_output2.squeeze(0)
        # # #现在可以把attn_output作为新的feature_1
        # feature_2 = attn_output2
        #

        # feature_3 = feature_3.unsqueeze(0)
        # # 把feature_1作为query,feature_2和feature3作为key和value
        # attn_output3, _ = self.attention_3(feature_3, feature_3, feature_3)
        # attn_output3 = attn_output3.squeeze(0)
        # # #现在可以把attn_output作为新的feature_1
        # feature_3 = attn_output3
        # feature_3 = self.l1(feature_3)  # 使用线性层l1对feature3进行变换
        # #
        # feature_5 = feature_5.unsqueeze(0)
        # # 把feature_1作为query,feature_2和feature3作为key和value
        # attn_output5, _ = self.attention_5(feature_5, feature_5, feature_5)
        # attn_output5 = attn_output5.squeeze(0)
        # # #现在可以把attn_output作为新的feature_1
        # feature_5 = attn_output5
        # feature_5 = self.l3(feature_5)  # 使用线性层l3对feature5进行变换

        #
        # feature_6 = feature_6.unsqueeze(0)
        # # 把feature_1作为query,feature_2和feature3作为key和value
        # attn_output6 = self.attention(feature_6, feature_6, feature_6)
        # attn_output6 = attn_output6.squeeze(0)
        # # #现在可以把attn_output作为新的feature_1
        # feature_6 = attn_output6
        #
        # # #多头注意力处理
        # #为了使用MultiheadAttention,特征需要额外的序列长度维度，假设为1
        # feature_7 = feature_7.unsqueeze(0)
        #  #把feature_1作为query,feature_2和feature3作为key和value
        # attn_output7,_ = self.attention_7(feature_7,feature_7,feature_7)
        # attn_output7 = attn_output7.squeeze(0)
        # # #现在可以把attn_output作为新的feature_1
        # feature_7 = attn_output7
        # feature_7 = self.l2(feature_7)  # 使用线性层l2对feature7进行变换
        # #
        #
        # feature_8 = feature_8.unsqueeze(0)
        # # 把feature_1作为query,feature_2和feature3作为key和value
        # attn_output8, _ = self.attention_8(feature_8, feature_8, feature_8)
        # attn_output8 = attn_output8.squeeze(0)
        # # #现在可以把attn_output作为新的feature_1
        # feature_8 = attn_output8
        # feature_8 = self.l4(feature_8)  # 使用线性层l4对feature8进行变换

        """对使用CNN的feature_3和feature_7进行特征融合"""
        # feature_3 = feature_3.unsqueeze(0)
        # # print("feature_3的unsqueeze完的形状", feature_3.shape)
        # feature_7 = feature_7.unsqueeze(0)
        # # print("feature_7的unsqueeze完的形状", feature_7.shape)
        # # 应用多头注意力，这里假设 feature_3 为 query，feature_7 为 key 和 value
        # attn_output,_ = self.attention(feature_3, feature_7, feature_7)
        # attn_output = attn_output.squeeze(0)
        # feature_3_7 = attn_output
        # print("feature_3_7的squeeze完的形状", feature_3_7.shape)
        # #返回结果
        # #
        # # """"对使用FCN的特征feature_1,feature_2,feature_5,feature_6特征融合"""
        # feature_1 = feature_1.unsqueeze(0)
        # # print("feature_1的unsqueeze完的形状", feature_1.shape)
        # feature_2 = feature_2.unsqueeze(0)
        # # print("feature_2的unsqueeze完的形状", feature_2.shape)
        # feature_5 = feature_5.unsqueeze(0)
        # # print("feature_5的unsqueeze完的形状", feature_5.shape)
        # feature_6 = feature_6.unsqueeze(0)
        # # print("feature_6的unsqueeze完的形状", feature_6.shape)
        # query1 = feature_1
        # key1 = torch.cat([feature_2,feature_5,feature_6])
        # value1 =key1
        # atten_output,_ = self.attention(query1,key1,value1)
        # atten_output = atten_output.squeeze(0)
        # feature_1_2_5_6 = atten_output
        # print("feature_1_2_5_6的squeeze完的形状", feature_1_2_5_6.shape)

        """对feature_8进行自注意力"""
        # feature_8 = feature_8.unsqueeze(0)
        # # print("feature_8的unsqueeze完的形状", feature_8.shape)
        # # 应用多头注意力，这里假设 feature_3 为 query，feature_7 为 key 和 value
        # attn_output_8,_ = self.attention_8(feature_8, feature_8, feature_8)
        # attn_output_8 = attn_output_8.squeeze(0)
        # feature_8 = attn_output_8
        # print("feature_8的squeeze完的形状", feature_8.shape)

        """对context_depend的feeature_1,2,3进行特征融合"""
        # feature_1 = feature_1.unsqueeze(0)
        # feature_2 = feature_2.unsqueeze(0)
        # feature_3 = feature_3.unsqueeze(0)
        # atten_output,_ = self.attention(feature_1,feature_2,feature_3)
        # atten_output = atten_output.squeeze(0)
        # feature_1_2_3 = atten_output
        #
        #
        """对context_free的feature_5,6,7,8进行融合"""
        # feature_5 = feature_5.unsqueeze(0)
        # feature_6 = feature_6.unsqueeze(0)
        # feature_7 = feature_7.unsqueeze(0)
        # feature_8 = feature_8.unsqueeze(0)
        # query1 = feature_5
        # key1 = torch.cat([feature_6,feature_7,feature_8])
        # value1 =key1
        # atten_output1,_=self.attention(query1,key1,value1)
        # atten_output1 = atten_output1.squeeze(0)
        # feature_5_6_7_8 = atten_output1

        # all_features = [feature_1_2_5_6,feature_3_7,feature_8]  # 组合所有特征
        # all_features = [feature_1_2_3,feature_5_6_7_8]  # 组合所有特征
        # all_features = [feature_1, feature_2,feature_3,feature_5,feature_6,feature_7,feature_8] #组合所有特征
        # 经过变换和没经过变换的都放在all_feature列表中
        # if self.ablation_study: #切除研究法，就是一种在模型中移除某些部分以观察其对性能影响的方法
        #     #如果self.ablation_study为True,则根据variant_to_drop的值来决定哪些特征要被移除
        #     #如果variant_to_drop是[1,3]，那么feature_1和feature_3将不会被包括在all_features中
        #     tmp = all_features #把all_features列表赋值给tmp
        #     all_features = []  #把all_features置为空
        #     drop = [] #创建一个drop列表，来存储哪些feature被移除了，如果没有被移除就添加上Flase,被移除了就添加上True
        #     drop.append(True) if 1 in self.variant_to_drop else drop.append(False)
        #     drop.append(True) if 2 in self.variant_to_drop else drop.append(False)
        #     drop.append(True) if 3 in self.variant_to_drop else drop.append(False)
        #     drop.append(True) if 5 in self.variant_to_drop else drop.append(False)
        #     drop.append(True) if 6 in self.variant_to_drop else drop.append(False)
        #     drop.append(True) if 7 in self.variant_to_drop else drop.append(False)
        #     drop.append(True) if 8 in self.variant_to_drop else drop.append(False)
        #     for i in range(len(drop)): #遍历drop列表中的每一项
        #         if not drop[i]: #如果没有被移除
        #             all_features.append(tmp[i])#就将其添加到all_feature中
        # # feature_list = torch.cat(all_features, axis=1) #将all_features列表中的所有张量按第二个维度拼接起来
        # x = self.drop_out(feature_list) #将dropout应用于连接起来的特征
        # print("链接起来的特征形状：",x.shape)
        # x = self.l5(x) #使用线性层l5进行进一步的特征变换
        # # print("l5线性变化后得到的x形状", x.shape)
        # # just for pca analysis only
        # pca_features = x #将处理后的x赋值给pca,这代表只用于pca analysis
        #
        # x = self.relu(x) #使用ReLu激活函数对特征进行非线性变换
        # x = self.drop_out(x) #应用dropout
        # x = self.out_proj(x) #使用out_proj线性层得到模型的最终输出
        #
        # if not need_features: #如果不需要特征
        #     return x #直接输出变化后的结果
        # else:
        #     return x, pca_features #把输出后的结果和pca_features一起输出


class EnsemblePCAModel(nn.Module):
    def __init__(self, feature_dim):
        super(EnsemblePCAModel, self).__init__()
        self.FEATURE_DIM = feature_dim
        self.HIDDEN_DIM_DROPOUT_PROB = 0.3
        self.NUMBER_OF_LABELS = 2

        self.linear = nn.Linear(self.FEATURE_DIM, self.FEATURE_DIM)
        self.relu = nn.ReLU()
        self.drop_out = nn.Dropout(self.HIDDEN_DIM_DROPOUT_PROB)
        self.out_proj = nn.Linear(self.FEATURE_DIM, self.NUMBER_OF_LABELS)

    def forward(self, features, need_features=False):
        x = self.linear(features)
        x = self.relu(x)
        x = self.drop_out(x)
        x = self.out_proj(x)

        if not need_features:
            return x
        else:
            return x


class EnsembleModelHunkLevelFCN(nn.Module):
    def __init__(self, ablation_study=False, variant_to_drop=None):
        super(EnsembleModelHunkLevelFCN, self).__init__()
        self.FEATURE_DIM = 768
        self.DENSE_DIM = 128
        self.CNN_FEATURE_DIM = 300
        self.HIDDEN_DIM_DROPOUT_PROB = 0.3
        self.NUMBER_OF_LABELS = 2
        # need 2 linear layer to project CNN feature dim to 768
        # 1 for variant 3
        # 1 for variant 7
        # self.l1 = nn.Linear(self.CNN_FEATURE_DIM, self.FEATURE_DIM)
        # self.l2 = nn.Linear(self.CNN_FEATURE_DIM * 2, self.FEATURE_DIM)

        # need 1 linear layer to project variant 5 feature to 768

        self.l3 = nn.Linear(self.DENSE_DIM, self.FEATURE_DIM)

        # need 1 linear layer to project variant 8 feature to 768
        self.l4 = nn.Linear(self.DENSE_DIM, self.FEATURE_DIM)

        # 1 layer to combine
        self.ablation_study = ablation_study

        if not self.ablation_study:
            self.l5 = nn.Linear(7 * self.FEATURE_DIM, self.FEATURE_DIM)
        else:
            self.l5 = nn.Linear((7 - len(variant_to_drop)) * self.FEATURE_DIM, self.FEATURE_DIM)

        self.variant_to_drop = variant_to_drop

        self.relu = nn.ReLU()

        self.drop_out = nn.Dropout(self.HIDDEN_DIM_DROPOUT_PROB)
        self.out_proj = nn.Linear(self.FEATURE_DIM, self.NUMBER_OF_LABELS)

    def forward(self, feature_1, feature_2, feature_3, feature_5, feature_6, feature_7, feature_8, need_features=False):
        # feature_3 = self.l1(feature_3)
        # feature_7 = self.l2(feature_7)
        feature_5 = self.l3(feature_5)
        feature_8 = self.l4(feature_8)
        all_features = [feature_1, feature_2, feature_3, feature_5, feature_6, feature_7, feature_8]
        if self.ablation_study:
            tmp = all_features
            all_features = []
            drop = []
            drop.append(True) if 1 in self.variant_to_drop else drop.append(False)
            drop.append(True) if 2 in self.variant_to_drop else drop.append(False)
            drop.append(True) if 3 in self.variant_to_drop else drop.append(False)
            drop.append(True) if 5 in self.variant_to_drop else drop.append(False)
            drop.append(True) if 6 in self.variant_to_drop else drop.append(False)
            drop.append(True) if 7 in self.variant_to_drop else drop.append(False)
            drop.append(True) if 8 in self.variant_to_drop else drop.append(False)
            for i in range(len(drop)):
                if not drop[i]:
                    all_features.append(tmp[i])
        feature_list = torch.cat(all_features, axis=1)
        x = self.drop_out(feature_list)
        x = self.l5(x)

        # just for pca analysis only
        pca_features = x

        x = self.relu(x)
        x = self.drop_out(x)
        x = self.out_proj(x)

        if not need_features:
            return x
        else:
            return x, pca_features


class EnsembleModelFileLevelCNN(nn.Module):
    def __init__(self, ablation_study=False, variant_to_drop=None):
        super(EnsembleModelFileLevelCNN, self).__init__()
        self.FEATURE_DIM = 768
        self.DENSE_DIM = 128
        self.CNN_FEATURE_DIM = 300
        self.HIDDEN_DIM_DROPOUT_PROB = 0.3
        self.NUMBER_OF_LABELS = 2
        # need 2 linear layer to project CNN feature dim to 768
        # 1 for variant 3
        # 1 for variant 7
        self.l1 = nn.Linear(self.CNN_FEATURE_DIM, self.FEATURE_DIM)
        self.l2 = nn.Linear(self.CNN_FEATURE_DIM * 2, self.FEATURE_DIM)

        # need 1 linear layer to project variant 5 feature to 768

        self.l3 = nn.Linear(self.DENSE_DIM, self.FEATURE_DIM)

        # need 1 linear layer to project variant 8 feature to 768
        self.l4 = nn.Linear(self.DENSE_DIM, self.FEATURE_DIM)

        # 1 layer to combine
        self.ablation_study = ablation_study

        if not self.ablation_study:
            self.l5 = nn.Linear(7 * self.FEATURE_DIM, self.FEATURE_DIM)
        else:
            self.l5 = nn.Linear((7 - len(variant_to_drop)) * self.FEATURE_DIM, self.FEATURE_DIM)

        # giang, need 2 more linear layer, each for variant 2 and variant 6.
        # self.l5 is already defined

        self.l6 = nn.Linear(self.CNN_FEATURE_DIM, self.FEATURE_DIM)
        self.l7 = nn.Linear(self.CNN_FEATURE_DIM * 2, self.FEATURE_DIM)

        self.variant_to_drop = variant_to_drop

        self.relu = nn.ReLU()

        self.drop_out = nn.Dropout(self.HIDDEN_DIM_DROPOUT_PROB)
        self.out_proj = nn.Linear(self.FEATURE_DIM, self.NUMBER_OF_LABELS)

    def forward(self, feature_1, feature_2, feature_3, feature_5, feature_6, feature_7, feature_8):
        feature_3 = self.l1(feature_3)
        feature_7 = self.l2(feature_7)
        feature_5 = self.l3(feature_5)
        feature_8 = self.l4(feature_8)
        feature_2 = self.l6(feature_2)
        feature_6 = self.l7(feature_6)
        all_features = [feature_1, feature_2, feature_3, feature_5, feature_6, feature_7, feature_8]
        if self.ablation_study:
            tmp = all_features
            all_features = []
            drop = []
            drop.append(True) if 1 in self.variant_to_drop else drop.append(False)
            drop.append(True) if 2 in self.variant_to_drop else drop.append(False)
            drop.append(True) if 3 in self.variant_to_drop else drop.append(False)
            drop.append(True) if 5 in self.variant_to_drop else drop.append(False)
            drop.append(True) if 6 in self.variant_to_drop else drop.append(False)
            drop.append(True) if 7 in self.variant_to_drop else drop.append(False)
            drop.append(True) if 8 in self.variant_to_drop else drop.append(False)
            for i in range(len(drop)):
                if not drop[i]:
                    all_features.append(tmp[i])
        feature_list = torch.cat(all_features, axis=1)
        x = self.drop_out(feature_list)
        x = self.l5(x)
        x = self.relu(x)
        x = self.drop_out(x)
        x = self.out_proj(x)

        return x

