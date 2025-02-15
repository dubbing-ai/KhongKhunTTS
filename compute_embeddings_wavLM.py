import argparse
import os
from argparse import RawTextHelpFormatter

import torch
import torchaudio
from tqdm import tqdm

from wavlm_module.WavLM import WavLM, WavLMConfig

from TTS.config import load_config
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.utils.managers import save_file

from typing import Callable, Union, List


def load_wavlm_model(model_path, use_cuda=True):
    device = 'cuda' if use_cuda and torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    checkpoint = torch.load(model_path, map_location=device)

    cfg = WavLMConfig(checkpoint['cfg'])
    model = WavLM(cfg).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    return model, device, cfg



def extract_wavlm_embedding(model, wav_file: Union[str, List[str]], cfg, use_cuda) -> list:
    """Compute a embedding from a given audio file.

    Args:
        wav_file (Union[str, List[str]]): Target file path.

    Returns:
        list: Computed embedding.
    """

    def _compute(wav_file: str):
        waveform, sampling_rate = torchaudio.load(wav_file)
        
        if torch.backends.mps.is_available():
            waveform = waveform.to('mps')
            
        if use_cuda:
            waveform = waveform.cuda()
        
        if sampling_rate != 16000:
            resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
            waveform = resampler(waveform)
            
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # extract the representation of last layer
        if cfg.normalize:
            waveform = torch.nn.functional.layer_norm(waveform , waveform.shape)
        rep = model.extract_features(waveform)[0]

        return rep.mean(dim=1)

    if isinstance(wav_file, list):
        # compute the mean embedding
        embeddings = None
        for wf in wav_file:
            embedding = _compute(wf)
            if embeddings is None:
                embeddings = embedding
            else:
                embeddings += embedding
        return (embeddings / len(wav_file))[0].tolist()
    embedding = _compute(wav_file)
    
    return embedding[0].tolist()


def compute_wavlm_embedding(wav_file: Union[str, List[str]], processor, model, device='cuda') -> list:
    """
    Compute WavLM embedding from a given audio file or list of audio files.
    Args:
        wav_file (Union[str, List[str]]): Target file path or list of file paths.
        processor: WavLM processor
        model: WavLM model
        device: Computing device ('cuda' or 'cpu')
    Returns:
        list: Computed embedding as a list of floats.
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
        
        # Process through WavLM
        inputs = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get the mean of last hidden states as the embedding
        embedding = torch.mean(outputs.last_hidden_state, dim=1)
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
        # Average and convert to list
        embedding = (embeddings / len(wav_file))
    else:
        # Single file processing
        embedding = _compute(wav_file)
    
    return embedding.tolist()


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
    wavlm_model, _, cfg = load_wavlm_model(model_path, use_cuda)

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
        embedding = extract_wavlm_embedding(wavlm_model, audio_file, cfg, use_cuda)

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
        help="Path to WavLM model checkpoint file.",
        default="https://huggingface.co/microsoft/wavlm-base-plus/resolve/main/wavlm_base_plus.pt",
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
