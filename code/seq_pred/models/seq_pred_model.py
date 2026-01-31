from misc.require_lib import *

class CTCBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, rnn_layers=2, dropout=0.3):
        super(CTCBaseline, self).__init__()
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=rnn_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if rnn_layers > 1 else 0
        )
        # num_classes + 1 for blank
        self.fc = nn.Linear(hidden_dim * 2, num_classes + 1)
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, features, feature_lengths):
        packed_input = nn.utils.rnn.pack_padded_sequence(
            features, feature_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.rnn(packed_input)
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True
        )
        logits = self.fc(output)
        log_probs = self.log_softmax(logits)
        return log_probs.transpose(0, 1), output_lengths
    
class SlowFastConformerCTC(nn.Module):
    def __init__(self, slowfast_dim, num_classes, conformer_dim=256, nhead=4, ffn_dim=1024, num_layers=8, dropout=0.1):

        super(SlowFastConformerCTC, self).__init__()


        self.input_projection = nn.Linear(slowfast_dim, conformer_dim)
        self.dropout = nn.Dropout(p=dropout)
        

        self.conformer = torchaudio.models.Conformer(
            input_dim=conformer_dim,
            num_heads=nhead,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            depthwise_conv_kernel_size=31,
            dropout=dropout
        )
        

        self.output_projection = nn.Linear(conformer_dim, num_classes + 1)
        

        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, features, feature_lengths):
        
        projected_features = self.input_projection(features)
        projected_features = self.dropout(projected_features)

        output, output_lengths = self.conformer(projected_features, feature_lengths)
        
        logits = self.output_projection(output)

        log_probs = self.log_softmax(logits)

        log_probs = log_probs.transpose(0, 1)
        
        return log_probs, output_lengths