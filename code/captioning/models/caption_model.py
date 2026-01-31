from misc.require_lib import *
from transformers import (
    BartConfig, 
    BartForCausalLM, 
    EncoderDecoderModel, 
    PreTrainedModel, 
    PretrainedConfig,
    AutoConfig,
    AutoModel
)
from transformers.modeling_outputs import BaseModelOutput 

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
    
AutoConfig.register("audio_encoder", AudioEncoderConfig)
AutoModel.register(AudioEncoderConfig, AudioEncoder)

def create_audio_bart_model(feature_dim=2304, bart_model_name="facebook/bart-base"):

    decoder_config = BartConfig.from_pretrained(bart_model_name)
    bart_hidden_size = decoder_config.hidden_size

    encoder_config = AudioEncoderConfig(feature_dim=feature_dim, hidden_size=bart_hidden_size)
    audio_encoder = AudioEncoder(encoder_config)

    model = EncoderDecoderModel.from_encoder_decoder_pretrained(
        encoder_pretrained_model_name_or_path=None,  
        decoder_pretrained_model_name_or_path=bart_model_name,
        encoder_model=audio_encoder 
    )

    model.config.decoder_start_token_id = model.config.decoder.bos_token_id
    model.config.eos_token_id = model.config.decoder.eos_token_id
    model.config.pad_token_id = model.config.decoder.pad_token_id
    
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.max_length = 256
    model.config.min_length = 20
    model.config.no_repeat_ngram_size = 3
    model.config.early_stopping = True
    model.config.length_penalty = 2.0
    model.config.num_beams = 4
    
    return model
  
