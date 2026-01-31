from misc.require_lib import *
from transformers import (
    BartConfig, 
    BartForConditionalGeneration, 
    PreTrainedModel, 
    PretrainedConfig
)

from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput

class AudioEncoderConfig(PretrainedConfig):
    model_type = "audio_encoder"
    def __init__(self, feature_dim=2304, hidden_size=768, nhead=8, num_encoder_layers=10, **kwargs):
        super().__init__(**kwargs)
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers

class AudioEncoder(PreTrainedModel):
    config_class = AudioEncoderConfig
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        hidden_size = config.hidden_size

        self.input_proj = nn.Linear(config.feature_dim, hidden_size)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=config.nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_encoder_layers)

    def forward(self, inputs_embeds, attention_mask=None, **kwargs):

        projected_features = self.input_proj(inputs_embeds) # (Batch, Time, d_model)
        
        if attention_mask is not None:
            padding_mask = (attention_mask == 0) 
        else:
            padding_mask = None

        encoded_features = self.transformer_encoder(projected_features, src_key_padding_mask=padding_mask)

        return BaseModelOutput(last_hidden_state=encoded_features)

class AudioTextBART(PreTrainedModel):

    config_class = BartConfig 
    
    def __init__(self, config: BartConfig, feature_dim=2304, num_encoder_layers=10):
        super().__init__(config)
        

        self.bart = BartForConditionalGeneration(config)
        

        audio_encoder_config = AudioEncoderConfig(
            feature_dim=feature_dim, 
            hidden_size=config.d_model,
            num_encoder_layers=num_encoder_layers
        )
        self.audio_encoder = AudioEncoder(audio_encoder_config)

    def forward(
        self, 
        input_ids,              
        attention_mask,          
        audio_inputs_embeds,   
        audio_attention_mask,   
        labels=None,         
        **kwargs
    ):

        text_encoder_outputs = self.bart.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        text_hidden_states = text_encoder_outputs.last_hidden_state
        

        audio_encoder_outputs = self.audio_encoder(
            inputs_embeds=audio_inputs_embeds,
            attention_mask=audio_attention_mask
        )
        audio_hidden_states = audio_encoder_outputs.last_hidden_state


        combined_hidden_states = torch.cat([text_hidden_states, audio_hidden_states], dim=1)
        combined_attention_mask = torch.cat([attention_mask, audio_attention_mask], dim=1)


        fused_encoder_outputs = BaseModelOutput(
            last_hidden_state=combined_hidden_states,
            hidden_states=None,
            attentions=None
        )


        return self.bart(
            encoder_outputs=fused_encoder_outputs,
            attention_mask=combined_attention_mask,
            labels=labels,
            **kwargs
        )


    def generate(self, input_ids, attention_mask, audio_inputs_embeds, audio_attention_mask, **generate_kwargs):

        
        text_encoder_outputs = self.bart.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        text_hidden_states = text_encoder_outputs.last_hidden_state
        
        audio_encoder_outputs = self.audio_encoder(
            inputs_embeds=audio_inputs_embeds,
            attention_mask=audio_attention_mask
        )
        audio_hidden_states = audio_encoder_outputs.last_hidden_state

        combined_hidden_states = torch.cat([text_hidden_states, audio_hidden_states], dim=1)
        combined_attention_mask = torch.cat([attention_mask, audio_attention_mask], dim=1)
        
        fused_encoder_outputs = BaseModelOutput(
            last_hidden_state=combined_hidden_states
        )
        

        return self.bart.generate(
            encoder_outputs=fused_encoder_outputs,
            attention_mask=combined_attention_mask,
            **generate_kwargs
        )

def create_audio_text_bart_model(bart_model_name="facebook/bart-base", feature_dim=2304):

    config = BartConfig.from_pretrained(bart_model_name)

    model = AudioTextBART(config, feature_dim=feature_dim)
    

    model.config.max_length = 512
    model.config.min_length = 20
    model.config.num_beams = 4
    model.config.length_penalty = 2.0
    model.config.no_repeat_ngram_size = 3
    
    return model