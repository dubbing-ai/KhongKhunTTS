import argparse
import os
from argparse import RawTextHelpFormatter

import torch
import torchaudio
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

from TTS.config import load_config
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.utils.managers import save_file

from typing import Callable, Union, List


def load_wavlm_model(model_path="microsoft/wavlm-base-plus-sv", use_cuda=True):
    """Load WavLM model and feature extractor from HuggingFace.
    
    Args:
        model_path (str): HuggingFace model path or local path
        use_cuda (bool): Whether to use CUDA if available
        
    Returns:
        tuple: (model, feature_extractor, device)
    """
    device = 'cuda' if use_cuda and torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    # Load model and feature extractor
    model = WavLMForXVector.from_pretrained(model_path).to(device)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
    
    model.eval()
    return model, feature_extractor, device


def extract_wavlm_embedding(model, feature_extractor, wav_file: Union[str, List[str]], device='cuda') -> list:
    """Compute a embedding from a given audio file using WavLM.

    Args:
        model: WavLM model
        feature_extractor: WavLM feature extractor
        wav_file (Union[str, List[str]]): Target file path or list of paths
        device (str): Computing device

    Returns:
        list: Computed embedding
    """
    def _compute(wav_file: str):
        # Load and resample audio if necessary
        waveform, sample_rate = torchaudio.load(wav_file)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Convert to numpy array for feature extractor
        audio_array = waveform.squeeze().numpy()
        
        # Process through WavLM
        inputs = feature_extractor(audio_array, sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Normalize embeddings
        embedding = torch.nn.functional.normalize(outputs.embeddings, dim=-1)
        return embedding

    if isinstance(wav_file, list):
        # Compute the mean embedding for multiple files
        embeddings = None
        for wf in wav_file:
            embedding = _compute(wf)
            if embeddings is None:
                embeddings = embedding
            else:
                embeddings += embedding
        return (embeddings / len(wav_file))[0].cpu().tolist()
    
    embedding = _compute(wav_file)
    return embedding[0].cpu().tolist()


def compute_embeddings(
    model_path,
    config_path,
    output_path,
    old_speakers_file=None,
    old_append=False,
    config_dataset_path=None,
    formatter_name=None,
    formatter: Callable = None,
    dataset_name=None,
    dataset_path=None,
    meta_file_train=None,
    meta_file_val=None,
    disable_cuda=False,
    no_eval=False,
):
    use_cuda = torch.cuda.is_available() and not disable_cuda

    if config_dataset_path is not None:
        c_dataset = load_config(config_dataset_path)
        meta_data_train, meta_data_eval = load_tts_samples(c_dataset.datasets, eval_split=not no_eval, formatter=formatter)
    else:
        c_dataset = BaseDatasetConfig()
        c_dataset.formatter = formatter_name
        c_dataset.dataset_name = dataset_name
        c_dataset.path = dataset_path
        if meta_file_train is not None:
            c_dataset.meta_file_train = meta_file_train
        if meta_file_val is not None:
            c_dataset.meta_file_val = meta_file_val
        meta_data_train, meta_data_eval = load_tts_samples(c_dataset, eval_split=not no_eval, formatter=formatter)

    if meta_data_eval is None:
        samples = meta_data_train
    else:
        samples = meta_data_train + meta_data_eval

    # Load WavLM Model
    model, feature_extractor, device = load_wavlm_model(model_path, use_cuda)

    # Load old speaker mappings if provided
    speaker_mapping = {}
    if old_speakers_file is not None and old_append:
        speaker_mapping = torch.load(old_speakers_file)

    for fields in tqdm(samples):
        class_name = fields["speaker_name"]
        audio_file = fields["audio_file"]
        embedding_key = fields["audio_unique_name"]

        if embedding_key in speaker_mapping:
            speaker_mapping[embedding_key]["name"] = class_name
            continue

        # Compute speaker embedding using WavLM
        embedding = extract_wavlm_embedding(model, feature_extractor, audio_file, device)

        speaker_mapping[embedding_key] = {
            "name": class_name,
            "embedding": embedding,
        }

    if speaker_mapping:
        # Save computed embeddings
        if os.path.isdir(output_path):
            mapping_file_path = os.path.join(output_path, "dvector.pth")
        else:
            mapping_file_path = output_path

        os.makedirs(os.path.dirname(mapping_file_path), exist_ok=True)
        save_file(speaker_mapping, mapping_file_path)
        print("Speaker embeddings saved at:", mapping_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute speaker embeddings using WavLM.",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to WavLM model or HuggingFace model name.",
        default="microsoft/wavlm-base-plus-sv",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        help="Not used for WavLM but kept for compatibility.",
        default=None,
    )
    parser.add_argument(
        "--config_dataset_path",
        type=str,
        help="Path to dataset config file.",
        default=None,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path for output embeddings file.",
        default="dvector.pth",
    )
    parser.add_argument(
        "--old_file",
        type=str,
        help="Existing embedding file to append new embeddings.",
        default=None,
    )
    parser.add_argument(
        "--old_append",
        help="Append new embeddings to old file.",
        default=False,
        action="store_true",
    )
    parser.add_argument("--disable_cuda", type=bool, help="Disable CUDA.", default=False)
    parser.add_argument("--no_eval", help="Disable evaluation split.", default=False, action="store_true")
    parser.add_argument("--formatter_name", type=str, help="Dataset formatter name.", default=None)
    parser.add_argument("--dataset_name", type=str, help="Dataset name.", default=None)
    parser.add_argument("--dataset_path", type=str, help="Dataset path.", default=None)
    parser.add_argument("--meta_file_train", type=str, help="Train meta file.", default=None)
    parser.add_argument("--meta_file_val", type=str, help="Eval meta file.", default=None)

    args = parser.parse_args()

    compute_embeddings(
        args.model_path,
        args.config_path,
        args.output_path,
        old_speakers_file=args.old_file,
        old_append=args.old_append,
        config_dataset_path=args.config_dataset_path,
        formatter_name=args.formatter_name,
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        meta_file_train=args.meta_file_train,
        meta_file_val=args.meta_file_val,
        disable_cuda=args.disable_cuda,
        no_eval=args.no_eval,
    )