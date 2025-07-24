import torch
import torch.nn as nn

class ConvGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvGRUCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        self.conv_gates = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                                    out_channels=2 * self.hidden_dim,
                                    kernel_size=self.kernel_size,
                                    padding=self.padding,
                                    bias=self.bias)

        self.conv_candidate = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                                        out_channels=self.hidden_dim,
                                        kernel_size=self.kernel_size,
                                        padding=self.padding,
                                        bias=self.bias)

    def forward(self, input_tensor, h_cur):
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv_gates(combined)

        gamma, beta = torch.split(combined_conv, self.hidden_dim, dim=1)
        reset_gate = torch.sigmoid(gamma)
        update_gate = torch.sigmoid(beta)

        combined_candidate = torch.cat([input_tensor, reset_gate * h_cur], dim=1)
        cnv_candidate = self.conv_candidate(combined_candidate)
        candidate = torch.tanh(cnv_candidate)

        h_next = update_gate * h_cur + (1 - update_gate) * candidate
        return h_next

class ConvGRU(nn.Module):
    def __init__(self, input_dim, hidden_dims, kernel_size, n_layers, bias=True):
        super(ConvGRU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.bias = bias

        cell_list = []
        for i in range(self.n_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dims[i-1]
            cell_list.append(ConvGRUCell(input_dim=cur_input_dim,
                                         hidden_dim=self.hidden_dims[i],
                                         kernel_size=self.kernel_size,
                                         bias=self.bias))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        if input_tensor.dim() == 5:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        seq_len, b, _, h, w = input_tensor.size()

        if hidden_state is None:
            hidden_state = [torch.zeros(b, self.hidden_dims[i], h, w).to(input_tensor.device) for i in range(self.n_layers)]

        last_state_list = []
        for t in range(seq_len):
            h_current_layer = input_tensor[t]
            for i in range(self.n_layers):
                h_current_layer = self.cell_list[i](h_current_layer, hidden_state[i])
                hidden_state[i] = h_current_layer
            last_state_list.append(h_current_layer)

        return hidden_state[-1]

class MultimodalConvGRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, kernel_size, n_layers, num_classes, embedding_dim, bias=True):
        super(MultimodalConvGRUClassifier, self).__init__()
        self.conv_gru = ConvGRU(input_dim, hidden_dims, kernel_size, n_layers, bias)
        
        # Embedding layer for the class input
        self.class_embedding = nn.Embedding(num_classes, embedding_dim)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # The input to the fully connected layer is the concatenation of ConvGRU output and class embedding
        self.fc = nn.Linear(hidden_dims[-1] + embedding_dim, num_classes)

    def forward(self, x_flow, x_class):
        # x_flow is (batch_size, sequence_length, channels, height, width)
        # x_class is (batch_size) with class indices
        
        # Process optical flow sequence
        conv_gru_output = self.conv_gru(x_flow) # (batch_size, hidden_dims[-1], H, W)
        pooled_output = self.avgpool(conv_gru_output) # (batch_size, hidden_dims[-1], 1, 1)
        flow_features = pooled_output.view(pooled_output.size(0), -1) # (batch_size, hidden_dims[-1])
        
        # Process class input
        class_embedded = self.class_embedding(x_class) # (batch_size, embedding_dim)
        
        # Concatenate the features
        combined_features = torch.cat((flow_features, class_embedded), dim=1)
        
        # Classifier
        logits = self.fc(combined_features) # (batch_size, num_classes)
        return logits
