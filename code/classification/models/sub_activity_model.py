
from misc.require_lib import *
from models.pooling import SelfAttenPoolingMask
from models.block import ConvBlock
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SubactivityNet(nn.Module):
    def __init__(self, slow_channels=2048, fast_channels=256, rnn_hidden_size=256, num_classes=14):
        super(SubactivityNet, self).__init__()
        self.unliearity = nn.ELU()
        self.dropout = nn.Dropout(p=0.5)
        self.rnn_hidden_size = rnn_hidden_size
        self.slow_conv_compress = ConvBlock(in_c=slow_channels, out_c=128, kernel=(25, 4), padding=0)
        self.fast_conv_compress = ConvBlock(in_c=fast_channels, out_c=128, kernel=(100, 4), padding=0)

        self.rnn = nn.GRU(input_size=rnn_hidden_size,
                          hidden_size=rnn_hidden_size,
                          num_layers=2,
                          bias=True,
                          batch_first=True,
                          dropout=0.2,
                          bidirectional=True)
        self.atten_pool = SelfAttenPoolingMask(input_size=rnn_hidden_size*2)
        self.linear1 = nn.Linear(rnn_hidden_size * 2, 512, bias=True)
        self.linear2 = nn.Linear(512, 512, bias=True)
    
        self.projection = nn.Linear(512, num_classes, bias=True)

    def forward(self, slow_input, fast_input, mask, lengths):
        """
        Args:
            slow_input (torch.Tensor): [B, T_max, 2048, 25, 4]
            fast_input (torch.Tensor): [B, T_max, 256, 100, 4]
            mask (torch.Tensor): [B, T_max]
        """
        B, T_max, _, _, _ = slow_input.shape

        slow_input = slow_input.view(B * T_max, 2048, 25, 4)
        fast_input = fast_input.view(B * T_max, 256, 100, 4)

        slow_f = self.slow_conv_compress(slow_input)
        fast_f = self.fast_conv_compress(fast_input)

        slow_f = slow_f.view(B, T_max, -1)
        fast_f = fast_f.view(B, T_max, -1)
        fused_features = torch.cat([slow_f, fast_f], dim=-1) # [batch, T_max, 256]

        packed = pack_padded_sequence(fused_features, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.rnn(packed)
        rnn_out, _ = pad_packed_sequence(packed_out, batch_first=True)  # [B, T_max, rnn_hidden*2]
        # rnn_out, _ = self.rnn(fused_features) # x: [batch, 100, 256], rnn_out: [batch, 100, 256]

        atten_out = self.atten_pool(rnn_out, mask)

        out = self.linear1(atten_out) # [batch, 512]256
        out = self.unliearity(out)
        embedding = self.linear2(out)
        out = self.dropout(embedding)
        logits = self.projection(out) # [batch, 14]

        return embedding, logits


if __name__ == '__main__':

    model = SubactivityNet()

    BATCH_SIZE = 5
    MAX_LEN = 15
    NUM_CLASSES = 14

    actual_lengths = [15, 8, 12, 5, 10]

    padded_slow_input = torch.randn(BATCH_SIZE, MAX_LEN, 2048, 25, 4)
    padded_fast_input = torch.randn(BATCH_SIZE, MAX_LEN, 256, 100, 4)

    mask = torch.zeros(BATCH_SIZE, MAX_LEN)

    model = SubactivityNet(num_classes=NUM_CLASSES)

    print("--- Input Shapes ---")
    print(f"Padded Slow Input: {padded_slow_input.shape}")
    print(f"Padded Fast Input: {padded_fast_input.shape}")
    print(f"Mask:              {mask.shape}")

    logits, embedding = model(padded_slow_input, padded_fast_input, mask)

    print("\n--- Output Shapes ---")
    print(f"Logits:    {logits.shape}")    # 应该为 [BATCH_SIZE, NUM_CLASSES] -> [5, 14]
    print(f"Embedding: {embedding.shape}") # 应该为 [BATCH_SIZE, 1024] -> [5, 1024]
