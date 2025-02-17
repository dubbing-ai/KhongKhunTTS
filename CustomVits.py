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
