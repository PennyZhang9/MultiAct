import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from misc.require_lib import *
from models.pooling import SelfAttenPoolingMask, SelfAttenPooling
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from models.block import ConvBlock


class ActivityNet(nn.Module):
    def __init__(self, subactivity_channels=512, rnn_hidden_size=256, num_classes=3):
        super(ActivityNet, self).__init__()
        self.unliearity = nn.ELU()
        self.dropout = nn.Dropout(p=0.5)
        self.rnn_hidden_size = rnn_hidden_size

        self.subactivity_conv_compress = nn.Conv1d(in_channels=subactivity_channels, 
                                                   out_channels=256,
                                                   kernel_size=1)

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

    def forward(self, subactivity_input, mask, lengths):
        """
        Args:
            subactivity_input (torch.Tensor): [B, T_max, 512], [1, 14, 512]
            mask (torch.Tensor): [B, T_max], [1, 14]
        """
        
        subactivity_f = subactivity_input.transpose(1, 2)
        subactivity_f = self.subactivity_conv_compress(subactivity_f)
        rnn_in = subactivity_f.transpose(1, 2)

        packed_input = pack_padded_sequence(rnn_in, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.rnn(packed_input)
        rnn_out, _ = pad_packed_sequence(packed_out, batch_first=True)  # [B, T_max, hidden*2]
        # rnn_out, _ = self.rnn(rnn_in) # x: [batch, 100, 512], rnn_out: [batch, 100, 512]

        atten_out = self.atten_pool(rnn_out, mask)

        out = self.linear1(atten_out) # [batch, 512]
        out = self.unliearity(out)
        embedding = self.linear2(out)
        out = self.dropout(embedding)
        logits = self.projection(out) # [batch, 14]

        return embedding, logits


class EventEncoder(nn.Module):
    def __init__(self, slow_channels=2048, fast_channels=256):
        super(EventEncoder, self).__init__()

        self.slow_conv_compress = ConvBlock(in_c=slow_channels, out_c=128, kernel=(25, 4), padding=0)
        self.fast_conv_compress = ConvBlock(in_c=fast_channels, out_c=128, kernel=(100, 4), padding=0)
        
    def forward(self, slow_input, fast_input):
        EVENT_NUM, _, _, _ = slow_input.shape
        slow_f = self.slow_conv_compress(slow_input) # [T, 128, 1, 1]
        fast_f = self.fast_conv_compress(fast_input) # [T, 128, 1, 1]
        
        slow_f = slow_f.view(EVENT_NUM, -1)
        fast_f = fast_f.view(EVENT_NUM, -1)
        
        fused_features = torch.cat([slow_f, fast_f], dim=-1) # [T, 256]
        
        return fused_features

class SubActivityEncoder(nn.Module):
    def __init__(self, rnn_hidden_size=256, num_classes=14):
        super(SubActivityEncoder, self).__init__()

        self.rnn = nn.GRU(input_size=rnn_hidden_size,
                          hidden_size=rnn_hidden_size,
                          num_layers=2,
                          bias=True,
                          batch_first=True,
                          dropout=0.2,
                          bidirectional=True)
        self.attention_pool = SelfAttenPoolingMask(input_size=rnn_hidden_size*2)
        self.linear1 = nn.Linear(rnn_hidden_size * 2, 512, bias=True)
        self.linear2 = nn.Linear(512, 512, bias=True)
        self.unliearity = nn.ELU()
        self.dropout = nn.Dropout(p=0.5)
        self.projection = nn.Linear(512, num_classes, bias=True)
    
    def forward(self, event_sequences, event_lengths):
        """
        Args:
            event_sequences (torch.Tensor): 填充过的事件嵌入序列, [N, E_max, D]
                                           N 是批次中的总子活动数量 (Total_Sub_Activities)
            event_lengths (torch.Tensor): 每个序列的真实长度, [N,]
        
        Returns:
            torch.Tensor: 每个子活动的聚合表征, [N, rnn_output_dim]
        """

        E_max = event_sequences.size(1)
        packed_input = pack_padded_sequence(
            event_sequences, 
            event_lengths.cpu(), # pack_padded_sequence 需要 CPU 上的长度张量
            batch_first=True, 
            enforce_sorted=False
        )
        packed_output, _ = self.rnn(packed_input)

        # 将输出解包并填充回原始长度
        gru_out, _ = pad_packed_sequence(
            packed_output, 
            batch_first=True, 
            total_length=E_max
        ) # -> [N, E_max, rnn_output_dim]

        mask = torch.arange(E_max, device=event_sequences.device)[None, :] < event_lengths[:, None]

        atten_out = self.attention_pool(gru_out, mask)
        out = self.linear1(atten_out) # [batch, 512]256
        out = self.unliearity(out)
        embedding = self.linear2(out)
        out = self.dropout(embedding)
        logits = self.projection(out)
        return embedding, logits

class CrossAttentionLayer(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, context, context_mask=None):
        q = self.query_proj(query)
        k = self.key_proj(context)
        v = self.value_proj(context)
        scores = torch.bmm(q, k.transpose(1, 2)) / (self.embed_dim ** 0.5)
        if context_mask is not None:
            scores.masked_fill_(context_mask.unsqueeze(1) == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.bmm(attn_weights, v)
        return output

class SubActivityRefiner(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.cross_attention = CrossAttentionLayer(embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, sub_activity_embs, event_embs, event_mask):
        B_S, _, D = sub_activity_embs.shape
        _, E_max, _ = event_embs.shape
        
        # Cross Attention (Query, Context, Mask)
        context_vector = self.cross_attention(sub_activity_embs, event_embs, event_mask)
        
        # Residual connection and normalization
        refined_flat = self.layer_norm(sub_activity_embs.squeeze(1) + context_vector.squeeze(1))
        return refined_flat



class HierarchicalActivityNet(nn.Module):
    def __init__(self, slow_channels=2048, fast_channels=256, subactivity_embed=512, rnn_hidden_size=256, num_activities_classes=3, num_subactivities_classes=14):
        super(HierarchicalActivityNet, self).__init__()

        # subactvitty modeling
        self.event_encoder = EventEncoder(slow_channels=slow_channels, fast_channels=fast_channels)
        self.subactivity_encoder = SubActivityEncoder(num_classes=num_subactivities_classes)

        self.query_projection = nn.Linear(subactivity_embed, 256)

        self.sub_activity_refiner = SubActivityRefiner(256)
        

        # activity modeling
        self.activity_rnn = nn.GRU(input_size=rnn_hidden_size,
                                   hidden_size=rnn_hidden_size,
                                   num_layers=2,
                                   bias=True,
                                   batch_first=True,
                                   dropout=0.2,
                                   bidirectional=True)

        self.activity_attention_pool = SelfAttenPoolingMask(input_size=rnn_hidden_size * 2)

        self.linear1 = nn.Linear(rnn_hidden_size * 2, 512, bias=True)
        self.linear2 = nn.Linear(512, 512, bias=True)

        self.unliearity = nn.ELU()
        self.dropout = nn.Dropout(p=0.5)
        self.projection = nn.Linear(512, num_activities_classes, bias=True)

    def forward(self, batch):
        
        event_features = batch['event_features']
        structure = batch['structure']
        device = self.projection.weight.device

        event_embs = self.event_encoder(event_features['slow'].to(device), event_features['fast'].to(device)) # [E_NUM, 256]
        event_embs_per_sub = torch.split(event_embs, structure['event_lengths'].tolist()) # [16]
        padded_event_embs = pad_sequence(event_embs_per_sub, batch_first=True) # -> [Total_Sub, E_max, D]
        subactivity_embs, subactivity_logits = self.subactivity_encoder(
            padded_event_embs, 
            structure['event_lengths'].to(device)
        ) # -> [Total_Sub, rnn_hidden_dim * 2]
        

        query_source = subactivity_embs
        dynamic_query = self.query_projection(query_source)

        context = padded_event_embs # [16, 26, 256]
    
        E_max = padded_event_embs.size(1)
        event_mask = torch.arange(E_max, device=device)[None, :] < structure['event_lengths'].to(device)[:, None]

        refined_sub_embs_flat = self.sub_activity_refiner(
            dynamic_query.unsqueeze(1), # The new dynamic query
            context,                    # The context
            event_mask
        ) # -> [Total_Sub_Activities, D] [16, 256]

        refined_embs_per_main = torch.split(refined_sub_embs_flat, structure['sub_lengths'].tolist())
        padded_refined_embs = pad_sequence(refined_embs_per_main, batch_first=True)
        
        S_max = padded_refined_embs.size(1)
        activity_mask = torch.arange(S_max, device=device)[None, :] < structure['sub_lengths'].to(device)[:, None]
        
        # Use packed sequence for efficiency
        packed_input = pack_padded_sequence(padded_refined_embs, structure['sub_lengths'].cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.activity_rnn(packed_input) # 
        rnn_out, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=S_max) # [2, 14, 512]

        atten_out = self.activity_attention_pool(rnn_out, activity_mask)
        
        out = self.linear1(atten_out) # [batch, 512]256
        out = self.unliearity(out)
        embedding = self.linear2(out)
        out = self.dropout(embedding)
        activity_logits = self.projection(out) # [batch, 14]

        return {
            'main_logits': activity_logits, # [2, 3]
            'sub_logits': subactivity_logits # [16, 14] 并且要验证是否要平铺输出？
        }


if __name__ == '__main__':

    model = ActivityNet()

    BATCH_SIZE = 5
    MAX_LEN = 15
    NUM_CLASSES = 14

    actual_lengths = [15, 8, 12, 5, 10]

    padded_slow_input = torch.randn(BATCH_SIZE, MAX_LEN, 2048, 25, 4)
    padded_fast_input = torch.randn(BATCH_SIZE, MAX_LEN, 256, 100, 4)

    mask = torch.zeros(BATCH_SIZE, MAX_LEN)

    model = ActivityNet(num_classes=NUM_CLASSES)

    print("--- Input Shapes ---")
    print(f"Padded Slow Input: {padded_slow_input.shape}")
    print(f"Padded Fast Input: {padded_fast_input.shape}")
    print(f"Mask:              {mask.shape}")

    logits, embedding = model(padded_slow_input, padded_fast_input, mask)

    print("\n--- Output Shapes ---")
    print(f"Logits:    {logits.shape}")    # 应该为 [BATCH_SIZE, NUM_CLASSES] -> [5, 14]
    print(f"Embedding: {embedding.shape}") # 应该为 [BATCH_SIZE, 1024] -> [5, 1024]
