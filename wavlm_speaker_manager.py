from typing import Union, List, Any, Dict
import torch
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
import torch.nn.functional as F
import os

from TTS.tts.utils.speakers import SpeakerManager

class WavLMConfig:
    """Mock config class to match what VITS expects from speaker encoder."""
    def __init__(self, hidden_size: int = 512):
        self.hidden_size = hidden_size

class WavLMWrapper(torch.nn.Module):
    """Wrapper for WavLM model to handle l2_norm argument."""
    def __init__(self, model_name: str = "microsoft/wavlm-base-plus-sv", device: str = "cuda"):
        super().__init__()
        self.device = device
        # Initialize WavLM model and processor
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = WavLMForXVector.from_pretrained(model_name).to(device)
        self.model.eval()

        # Add mock config to match VITS expectations
        self.config = WavLMConfig(hidden_size=512)
        
    def forward(self, waveform: torch.Tensor, l2_norm: bool = False) -> torch.Tensor:
        """Forward pass with optional L2 normalization.
        
        Args:
            waveform (torch.Tensor): Input waveform tensor
            l2_norm (bool, optional): Whether to L2 normalize output. Defaults to False.
            
        Returns:
            torch.Tensor: Speaker embedding
        """
        # Prepare inputs
        if waveform.dim() == 2 and waveform.size(0) > 1:
            # Average multiple channels to mono
            waveform = waveform.mean(dim=0, keepdim=True)
        elif waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            
        # Convert to numpy for feature extractor
        audio_array = waveform.detach().cpu().numpy().squeeze()
        
        # Process through feature extractor
        inputs = self.feature_extractor(audio_array, sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.embeddings
            
        # Normalize if requested
        if l2_norm:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
            
        return embeddings


class WavLMXVectorManager(SpeakerManager):
    """Speaker manager that uses WavLM XVector while maintaining exact compatibility 
    with the standard SpeakerManager interface."""
    
    def __init__(
        self,
        data_items: List[List[Any]] = None,
        d_vectors_file_path: str = "",
        speaker_id_file_path: str = "",
        model_name: str = "microsoft/wavlm-base-plus-sv",
        use_cuda: bool = False
    ):
        super().__init__(
            data_items=data_items,
            d_vectors_file_path=d_vectors_file_path,
            speaker_id_file_path=speaker_id_file_path,
            use_cuda=use_cuda
        )
        
        self.device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        
        # Initialize WavLM model with wrapper
        self.encoder = WavLMWrapper(model_name, self.device)

        # Load d-vectors if provided
        if d_vectors_file_path and os.path.isfile(d_vectors_file_path):
            self.load_embeddings_from_file(d_vectors_file_path)
            print(f" > Loaded {len(self.name_to_id)} speakers from d-vectors file")
        
        # Initialize speaker mapping from data_items if provided and d-vectors weren't loaded
        if data_items and not hasattr(self, 'embeddings'):
            self.set_ids_from_data(data_items, parse_key="speaker_name")
            print(f" > Initialized {len(self.name_to_id)} speakers from data items")

    def load_embeddings_from_file(self, embedding_file_path: str):
        """Load speaker embeddings from file, matching the format expected by VITS."""
        self.embeddings = torch.load(embedding_file_path)
        
        # Create name-to-id mapping
        speaker_names = sorted(set(x["name"] for x in self.embeddings.values()))
        self.name_to_id = {name: idx for idx, name in enumerate(speaker_names)}
        self.id_to_name = {idx: name for idx, name in enumerate(speaker_names)}
        
        # Create embedding lookup by clip and name
        self.embeddings_by_names = {}
        for clip_data in self.embeddings.values():
            name = clip_data["name"]
            if name not in self.embeddings_by_names:
                self.embeddings_by_names[name] = []
            self.embeddings_by_names[name].append(clip_data["embedding"])

    def get_speakers(self) -> List[str]:
        """Get list of all speaker names."""
        return sorted(self.name_to_id.keys())

    @property
    def num_speakers(self):
        """Get number of speakers."""
        return len(self.name_to_id)

    @property
    def embedding_dim(self):
        """Get embedding dimension."""
        if hasattr(self, 'embeddings') and self.embeddings:
            first_embedding = next(iter(self.embeddings.values()))["embedding"]
            return len(first_embedding)
        return 512  # Default WavLM embedding size

    def get_embedding_by_clip(self, clip_idx: str) -> List:
        """Get embedding for a specific clip."""
        if hasattr(self, 'embeddings') and clip_idx in self.embeddings:
            return self.embeddings[clip_idx]["embedding"]
        return None

    def get_embeddings_by_name(self, name: str) -> List[List]:
        """Get all embeddings for a speaker name."""
        if hasattr(self, 'embeddings_by_names'):
            return self.embeddings_by_names.get(name, [])
        return []

    @staticmethod
    def init_from_config(config: "Coqpit", samples: Union[List[List], List[Dict]] = None) -> "WavLMXVectorManager":
        """Initialize from config, matching the standard SpeakerManager interface."""
        # Get d-vector file path from config
        d_vector_file = None
        if hasattr(config, "model_args") and hasattr(config.model_args, "d_vector_file"):
            if isinstance(config.model_args.d_vector_file, list):
                d_vector_file = config.model_args.d_vector_file[0]
            else:
                d_vector_file = config.model_args.d_vector_file

        # Define default speaker encoder config
        use_cuda = True
        if hasattr(config, "model_args") and hasattr(config.model_args, "speaker_encoder_config"):
            use_cuda = config.model_args.speaker_encoder_config.get("use_cuda", True)

        # Initialize the manager
        speaker_manager = WavLMXVectorManager(
            data_items=samples,
            d_vectors_file_path=d_vector_file,
            model_name="microsoft/wavlm-base-plus-sv",
            use_cuda=use_cuda
        )
        
        return speaker_manager