from typing import Dict, List, Union
import torch
import torch.nn.functional as F
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.languages import LanguageManager
from TTS.tts.models.vits import Vits

from coqpit import Coqpit

from wavlm_speaker_manager import WavLMXVectorManager

class CustomVits(Vits):
    """Custom VITS model that uses WavLM speaker manager"""

    def __init__(
        self,
        config: Coqpit,
        ap: "AudioProcessor" = None,
        tokenizer: "TTSTokenizer" = None,
        speaker_manager: SpeakerManager = None,
        language_manager: LanguageManager = None,
    ):
        super().__init__(config, ap, tokenizer, speaker_manager, language_manager)
        
        # Initialize speaker embedding dimension
        if hasattr(self.speaker_manager, "encoder"):
            self.embedded_speaker_dim = self.speaker_manager.encoder.config.hidden_size
        else:
            self.embedded_speaker_dim = 512  # Default WavLM embedding size

    def _set_cond_input(self, aux_input):
        """Set the conditional inputs for the model based on aux_input.
        
        Args:
            aux_input (dict): Auxiliary inputs containing d_vectors, speaker_ids, and language_ids.
            
        Returns:
            tuple: (speaker_ids, d_vectors, language_ids, aux_input)
        """
        sid, g, lid = None, None, None
        
        if aux_input is None:
            aux_input = {"d_vectors": None, "speaker_ids": None, "language_ids": None}
            
        # Handle speaker conditioning
        if aux_input.get("d_vectors") is not None:
            g = F.normalize(aux_input["d_vectors"]).unsqueeze(-1)
        elif aux_input.get("speaker_ids") is not None and self.args.use_speaker_embedding:
            sid = aux_input["speaker_ids"]
            g = self.emb_g(sid).unsqueeze(-1)
            
        # Ensure g is not None
        if g is None and hasattr(self, "emb_g"):
            # Use a default speaker if none provided
            g = self.emb_g(torch.zeros(1, dtype=torch.long, device=self.device)).unsqueeze(-1)
            
        # Handle language conditioning
        if aux_input.get("language_ids") is not None:
            lid = aux_input["language_ids"]
            
        return sid, g, lid, aux_input

    @staticmethod
    def init_from_config(config: "VitsConfig", samples: Union[List[List], List[Dict]] = None, verbose=True):
        """Initialize model from config"""
        # Validate upsampling rates
        upsample_rate = torch.prod(torch.as_tensor(config.model_args.upsample_rates_decoder)).item()

        if not config.model_args.encoder_sample_rate:
            assert (
                upsample_rate == config.audio.hop_length
            ), f" [!] Product of upsample rates must be equal to the hop length - {upsample_rate} vs {config.audio.hop_length}"
        else:
            encoder_to_vocoder_upsampling_factor = config.audio.sample_rate / config.model_args.encoder_sample_rate
            effective_hop_length = config.audio.hop_length * encoder_to_vocoder_upsampling_factor
            assert (
                upsample_rate == effective_hop_length
            ), f" [!] Product of upsample rates must be equal to the hop length - {upsample_rate} vs {effective_hop_length}"

        ap = AudioProcessor.init_from_config(config, verbose=verbose)
        tokenizer, new_config = TTSTokenizer.init_from_config(config)
        
        # Use WavLMXVectorManager instead of default SpeakerManager
        speaker_manager = WavLMXVectorManager.init_from_config(config, samples)
        
        language_manager = LanguageManager.init_from_config(config)

        return CustomVits(new_config, ap, tokenizer, speaker_manager, language_manager)
