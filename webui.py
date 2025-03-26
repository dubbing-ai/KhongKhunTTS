import os
import sys
import gradio as gr
import torch
from dotenv import load_dotenv
import string
from datetime import datetime
import subprocess
from pathlib import Path
import shutil

# Import from the transcriber module
from transcriber import transcribe_audio_from_path

# Try to import TTS modules
try:
    from TTS.tts.utils.synthesis import synthesis
    from TTS.utils.audio import AudioProcessor
    from TTS.tts.models import setup_model
    from TTS.config import load_config
    from TTS.tts.models.vits import *
    from TTS.tts.utils.speakers import SpeakerManager
    from TTS.bin.resample import resample_files
    from TTS.utils.vad import get_vad_model_and_utils, remove_silence
    from pydub import AudioSegment
except ImportError as e:
    print(f"Error importing TTS modules: {e}")
    print("Please install the required packages first")
    sys.exit(1)

# Constants
OUT_PATH = 'output'

# These will be populated by model selection
BASE_MODEL_PATH = None
MODEL_PATH = None
CONFIG_PATH = None
TTS_LANGUAGES = None

# Check if CUDA is available
USE_CUDA = torch.cuda.is_available()
print(f"CUDA available: {USE_CUDA}")

def scan_models():
    """Scan for available models and checkpoints"""
    checkpoint_info = {}
    model_base_dir = os.path.join(".", "model")
    
    # Check if model directory exists
    if not os.path.exists(model_base_dir):
        return checkpoint_info
    
    # Scan for model folders
    for model_dir in os.listdir(model_base_dir):
        full_model_path = os.path.join(model_base_dir, model_dir)
        
        # Skip if not a directory
        if not os.path.isdir(full_model_path):
            continue
        
        # Check if it contains config.json and language_ids.json
        if not os.path.exists(os.path.join(full_model_path, "config.json")) or \
           not os.path.exists(os.path.join(full_model_path, "language_ids.json")):
            continue
        
        # Find checkpoint files
        for file in os.listdir(full_model_path):
            if file.startswith("checkpoint_") and file.endswith(".pth"):
                # Store checkpoint with its parent folder
                checkpoint_info[file] = model_dir
    
    return checkpoint_info

def prepare_model_dropdown_options():
    """Create options for model dropdown in format suitable for UI"""
    checkpoint_info = scan_models()
    dropdown_options = []
    
    # Sort checkpoint names for better organization
    sorted_checkpoints = sorted(checkpoint_info.keys())
    
    for checkpoint in sorted_checkpoints:
        # Get the model folder for this checkpoint
        model_dir = checkpoint_info[checkpoint]
        # Format: model_name/checkpoint_name (showing both, but value is just checkpoint)
        option_text = f"{model_dir}/{checkpoint}"
        dropdown_options.append((option_text, checkpoint))
    
    return dropdown_options

def setup_api_key():
    """Check for API key in .env file or prompt user to enter it"""
    # Load from .env if exists
    if os.path.exists(".env"):
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            print("✓ API key loaded from .env file")
            return api_key
    
    # Prompt user for API key
    print("\nGEMINI API KEY is required for translation functionality.")
    api_key = input("Please enter your GEMINI API KEY (or press Enter to skip): ")
    
    if api_key:
        # Save to .env file
        with open(".env", "w") as f:
            f.write(f"GEMINI_API_KEY={api_key}")
        print("✓ API key saved to .env file")
        return api_key
    else:
        print("⚠️ No API key provided. Translation functionality will be disabled.")
        return None

def initialize_model(checkpoint_name):
    """Initialize the TTS model from a checkpoint name"""
    global BASE_MODEL_PATH, MODEL_PATH, CONFIG_PATH, TTS_LANGUAGES
    
    try:
        if not checkpoint_name:
            return None, None, None, None, "No checkpoint selected"
        
        # Get all available checkpoints with their folder info
        checkpoint_info = scan_models()
        
        # Check if the checkpoint exists in our scanned data
        if checkpoint_name not in checkpoint_info:
            return None, None, None, None, f"Checkpoint {checkpoint_name} not found"
        
        # Get the model directory for this checkpoint
        model_dir = checkpoint_info[checkpoint_name]
        
        # Set paths based on selected checkpoint
        BASE_MODEL_PATH = os.path.join(".", "model", model_dir)
        MODEL_PATH = os.path.join(BASE_MODEL_PATH, checkpoint_name)
        CONFIG_PATH = os.path.join(BASE_MODEL_PATH, "config.json")
        TTS_LANGUAGES = os.path.join(BASE_MODEL_PATH, "language_ids.json")
        
        # Check if model files exist
        if not os.path.exists(MODEL_PATH):
            return None, None, None, None, f"Model file not found: {MODEL_PATH}"
        if not os.path.exists(CONFIG_PATH):
            return None, None, None, None, f"Config file not found: {CONFIG_PATH}"
        if not os.path.exists(TTS_LANGUAGES):
            return None, None, None, None, f"Language file not found: {TTS_LANGUAGES}"
            
        print(f"Initializing model: {MODEL_PATH}")
        
        # Load the config
        C = load_config(CONFIG_PATH)
        
        # Load the audio processor
        ap = AudioProcessor(**C.audio)
        
        # Override config
        C["speakers_file"] = None
        C["d_vector_file"] = []
        C["language_ids_file"] = TTS_LANGUAGES
        
        C["model_args"]["speakers_file"] = None
        C["model_args"]["d_vector_file"] = []
        C["model_args"]["language_ids_file"] = TTS_LANGUAGES
        
        C.model_args['use_speaker_encoder_as_loss'] = False
        
        # Set up model
        model = setup_model(C)
        cp = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        
        # Remove speaker encoder from weights
        model_weights = cp['model'].copy()
        for key in list(model_weights.keys()):
            if "speaker_encoder" in key:
                del model_weights[key]
        
        model.load_state_dict(model_weights)
        model.eval()
        
        if USE_CUDA:
            model = model.cuda()
        
        # Create speaker manager
        SE_speaker_manager = SpeakerManager(
            encoder_model_path=C["model_args"]["speaker_encoder_model_path"],
            encoder_config_path=C["model_args"]["speaker_encoder_config_path"],
            use_cuda=USE_CUDA
        )
        
        # Set default inference parameters
        model.length_scale = 1.5
        model.inference_noise_scale = 0.2
        model.inference_noise_scale_dp = 0.2
        
        print("Model initialized successfully")
        return model, C, ap, SE_speaker_manager, None
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, None, None, f"Error initializing model: {str(e)}"

def process_reference_audio(audio_path):
    """Process reference audio for speaker embedding extraction"""
    try:
        print(f"Processing reference audio: {audio_path}")
        
        # Create a timestamp-based unique ID for this processing session
        process_id = datetime.now().strftime('%Y%m%d%H%M%S')
        
        # Handle gradio audio input (tuple of sample_rate and audio_data)
        if isinstance(audio_path, tuple):
            print("Processing audio from recording...")
            sample_rate, audio_data = audio_path
            temp_path = os.path.join("reference_voice", f"recording_{process_id}.wav")
            import scipy.io.wavfile as wav
            wav.write(temp_path, sample_rate, audio_data)
            audio_path = temp_path
            print(f"Saved recording to {temp_path}")
        
        if not os.path.exists(audio_path):
            return None, None, None, f"Audio file not found: {audio_path}"
        
        filename = os.path.basename(audio_path)
        basename = os.path.splitext(filename)[0]
        
        # Create speaker folder using the basename
        speaker_folder = basename
        speaker_dir = os.path.join("reference_voice", speaker_folder)
        os.makedirs(speaker_dir, exist_ok=True)
        
        # Define the reference WAV path
        ref_wav_path = os.path.join(speaker_dir, f"{basename}_reference.wav")
        
        # Copy or convert the file to the reference location
        if audio_path.endswith('.wav'):
            shutil.copy(audio_path, ref_wav_path)
            print(f"Copied WAV to reference path: {ref_wav_path}")
        else:
            try:
                # Convert non-WAV to WAV
                audio = AudioSegment.from_file(audio_path)
                audio.export(ref_wav_path, format="wav")
                print(f"Converted to WAV at reference path: {ref_wav_path}")
            except Exception as e:
                print(f"Error converting to WAV: {e}")
                return None, None, None, f"Error converting to WAV: {str(e)}"
        
        return audio_path, ref_wav_path, speaker_folder, None
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, None, f"Error processing reference audio: {str(e)}"

def process_input_audio(audio_path, model=None, C=None, ap=None, speaker_manager=None):
    """Process input audio file: convert to WAV, resample, normalize, and extract embedding"""
    try:
        print("Starting audio processing...")
        
        # Create a timestamp-based unique ID for this processing session
        process_id = datetime.now().strftime('%Y%m%d%H%M%S')
        
        # Create a unique directory for this file to avoid resampling the entire folder
        process_dir = os.path.join("./processed_audio", f"proc_{process_id}")
        os.makedirs(process_dir, exist_ok=True)
        print(f"Created process directory: {process_dir}")
        
        # Handle gradio audio input (tuple of sample_rate and audio_data)
        if isinstance(audio_path, tuple):
            print("Processing audio from recording...")
            sample_rate, audio_data = audio_path
            temp_path = os.path.join(process_dir, f"recording_{process_id}.wav")
            import scipy.io.wavfile as wav
            wav.write(temp_path, sample_rate, audio_data)
            audio_path = temp_path
            print(f"Saved recording to {temp_path}")
        else:
            print(f"Processing audio from file: {audio_path}")
        
        if not os.path.exists(audio_path):
            return None, None, None, f"Audio file not found: {audio_path}"
        
        filename = os.path.basename(audio_path)
        basename = os.path.splitext(filename)[0]
        extension = os.path.splitext(filename)[1].lower()
        
        # Processed audio path
        temp_wav_path = os.path.join(process_dir, f"{basename}_temp.wav")
        processed_wav_path = os.path.join(process_dir, f"{basename}_processed.wav")
        
        print(f"Converting to WAV format...")
        # Convert to WAV if needed
        if extension != ".wav":
            try:
                audio = AudioSegment.from_file(audio_path, format=extension[1:])
                audio.export(temp_wav_path, format="wav")
                print(f"Converted {extension} to WAV: {temp_wav_path}")
            except Exception as e:
                print(f"Error converting to WAV: {e}")
                return None, None, None, f"Error converting to WAV: {str(e)}"
        else:
            # Copy the file to process_dir
            try:
                shutil.copy(audio_path, temp_wav_path)
                print(f"Copied WAV to: {temp_wav_path}")
            except Exception as e:
                print(f"Error copying WAV file: {e}")
                return None, None, None, f"Error copying WAV file: {str(e)}"
            
        # Make sure CONFIG_PATH is available (model needs to be loaded)
        if not CONFIG_PATH or not os.path.exists(CONFIG_PATH):
            return None, None, None, "No model loaded. Please load a model first."
            
        # Get config for sample rate
        try:
            print("Loading model config...")
            local_C = C if C is not None else load_config(CONFIG_PATH)
            print(f"Model sample rate: {local_C.audio['sample_rate']}")
        except Exception as e:
            print(f"Error loading config: {e}")
            return None, None, None, f"Error loading config: {str(e)}"
        
        # Copy the temp file to processed path before resampling
        try:
            shutil.copy(temp_wav_path, processed_wav_path)
            print(f"Copied to processed path: {processed_wav_path}")
        except Exception as e:
            print(f"Error copying to processed path: {e}")
            return None, None, None, f"Error preparing for processing: {str(e)}"
        
        # Manually resample the file instead of using resample_files to avoid folder scanning
        try:
            print(f"Resampling file to {local_C.audio['sample_rate']}Hz...")
            import soundfile as sf
            
            # Read the audio file
            data, samplerate = sf.read(processed_wav_path)
            
            # Only resample if needed
            if samplerate != local_C.audio['sample_rate']:
                import librosa
                # Resample using librosa
                resampled_data = librosa.resample(
                    y=data.T if data.ndim > 1 else data, 
                    orig_sr=samplerate, 
                    target_sr=local_C.audio['sample_rate']
                )
                
                # Convert back to correct shape if stereo
                if data.ndim > 1:
                    resampled_data = resampled_data.T
                
                # Write resampled data
                sf.write(processed_wav_path, resampled_data, local_C.audio['sample_rate'])
                print(f"Resampled from {samplerate}Hz to {local_C.audio['sample_rate']}Hz")
            else:
                print("File already at target sample rate, skipping resampling")
        except Exception as e:
            print(f"Error during resampling: {e}")
            return None, None, None, f"Error during resampling: {str(e)}"
            
        # Trim silence
        try:
            print("Trimming silence...")
            model_and_utils = get_vad_model_and_utils(use_cuda=USE_CUDA, use_onnx=False)
            output_path, is_speech = remove_silence(
                model_and_utils,
                processed_wav_path,
                processed_wav_path,
                trim_just_beginning_and_end=True,
                use_cuda=USE_CUDA
            )
            print("Silence trimming complete")
        except Exception as e:
            print(f"Error trimming silence: {e}")
            # Don't return here, try to continue with normalization
        
        # Normalize audio using ffmpeg-normalize - fixing command arguments
        try:
            print("Normalizing audio...")
            normalize_cmd = [
                "ffmpeg-normalize", 
                processed_wav_path, 
                "-o", processed_wav_path, 
                "-nt", "rms", 
                "-t", "-27", 
                "-ar", str(local_C.audio['sample_rate']), 
                "-f"
            ]
            print(f"Running normalize command: {' '.join(normalize_cmd)}")
            
            result = subprocess.run(
                normalize_cmd, 
                check=True, 
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            print("Normalization complete")
            print(f"Normalize stdout: {result.stdout}")
        except subprocess.CalledProcessError as e:
            print(f"Error in ffmpeg-normalize: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            print("Continuing without normalization...")
        except FileNotFoundError:
            print("ffmpeg-normalize not found. Please install it or add it to your PATH.")
            print("Continuing without normalization...")
        except Exception as e:
            print(f"Unexpected error during normalization: {e}")
        
        # Extract embedding if speaker_manager is available
        embedding = None
        if speaker_manager:
            try:
                print("Extracting speaker embedding...")
                embedding = speaker_manager.compute_embedding_from_clip(processed_wav_path)
                if hasattr(embedding, 'shape'):
                    print(f"Embedding extraction complete: shape {embedding.shape}")
                else:
                    print(f"Embedding extraction complete: shape unknown")
            except Exception as e:
                print(f"Error extracting embedding: {e}")
                # Continue without embedding, will be computed later if needed
        else:
            print("Speaker manager not available, skipping embedding extraction")
        
        # Clean up temporary files
        try:
            if temp_wav_path != processed_wav_path and os.path.exists(temp_wav_path):
                os.remove(temp_wav_path)
                print(f"Removed temp file: {temp_wav_path}")
        except Exception as e:
            print(f"Error cleaning up temp files: {e}")
            # Not critical, continue
            
        print(f"Audio processing completed successfully: {processed_wav_path}")
        
        # Read the processed audio for display
        try:
            import soundfile as sf
            import numpy as np
            
            # Load the audio data
            audio_data, sample_rate = sf.read(processed_wav_path)
            
            # Convert to int16 for better compatibility with Gradio
            if audio_data.dtype != np.int16:
                audio_data = (audio_data * 32767).astype(np.int16)
            
            print(f"Loaded processed audio for display: {len(audio_data)} samples at {sample_rate}Hz, dtype: {audio_data.dtype}")
            
            # Return the audio data as tuple for Gradio display
            return (sample_rate, audio_data), processed_wav_path, embedding, None
        except Exception as e:
            print(f"Error reading processed audio: {e}")
            print(f"Returning path instead: {processed_wav_path}")
            # Fall back to returning path
            return processed_wav_path, processed_wav_path, embedding, None
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error processing audio: {e}")
        return None, None, None, f"Error processing audio: {str(e)}"

def transcribe_audio(audio_path, api_key):
    """Call the transcriber to process the audio"""
    try:
        # Process audio path if it's from gradio
        if isinstance(audio_path, tuple):
            sample_rate, audio_data = audio_path
            temp_path = f"temp_recording_{datetime.now().strftime('%Y%m%d%H%M%S')}.wav"
            import scipy.io.wavfile as wav
            wav.write(temp_path, sample_rate, audio_data)
            audio_path = temp_path
        
        print(f"Transcribing audio from: {audio_path}")
        transcription, translation = transcribe_audio_from_path(audio_path, api_key)
        
        # Clean up temp file if created
        if audio_path.startswith("temp_recording_") and os.path.exists(audio_path):
            os.remove(audio_path)
        
        if transcription and translation:
            transcription = transcription.strip('"')
            translation = translation.translate(str.maketrans('', '', string.punctuation)).strip('\n')
            return transcription, translation, ""
        else:
            return None, None, "Transcription failed. Please check the audio and try again."
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, f"Error during transcription: {str(e)}"

def synthesize_speech(reference_audio, translation_text, model, C, ap, speaker_manager, 
                     speaker_embedding=None, length_scale=1.5, noise_scale=0.2, noise_scale_dp=0.2):
    """Synthesize speech using the TTS model with adjustable parameters"""
    try:
        # Use provided speaker embedding if available, otherwise extract from reference audio
        reference_emb = speaker_embedding
        speaker_folder = None
        
        if reference_emb is None:
            # Process reference audio to get paths
            reference_path, reference_wav_path, speaker_folder, error = process_reference_audio(reference_audio)
            if error:
                return None, error
            
            if not reference_wav_path or not speaker_folder:
                return None, "Failed to process reference audio"
            
            print(f"Using reference wav: {reference_wav_path}")
            
            # Compute speaker embedding
            reference_emb = speaker_manager.compute_embedding_from_clip(reference_wav_path)
        else:
            print("Using pre-computed speaker embedding")
            # For output file naming, we need speaker_folder
            if isinstance(reference_audio, str):
                filename = os.path.basename(reference_audio)
                speaker_folder = os.path.splitext(filename)[0]
            else:
                speaker_folder = f"recording_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Set up model parameters for inference
        model.length_scale = length_scale  # Controls speech speed
        model.inference_noise_scale = noise_scale
        model.inference_noise_scale_dp = noise_scale_dp
        
        # Language ID (hardcoded to 0 for now, which is English according to the notebook)
        language_id = 0
        
        # Synthesize
        print(f"Synthesizing speech with text: {translation_text}")
        result = synthesis(
            model=model,
            text=translation_text,
            CONFIG=C,
            use_cuda=USE_CUDA,
            d_vector=reference_emb,
            style_wav=None,
            language_id=language_id,
            use_griffin_lim=True,
            do_trim_silence=False,
        )
        
        wav = result["wav"]
        
        # Convert to int16 to avoid Gradio warnings
        import numpy as np
        if wav.dtype == np.float32 or wav.dtype == np.float64:
            wav = (wav * 32767).astype(np.int16)
        
        # Save the synthesized audio
        model_name = os.path.basename(MODEL_PATH).rstrip('.pth')
        current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        
        # Create output directory structure
        output_dir = os.path.join(OUT_PATH, speaker_folder)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save file
        output_file = f"{model_name}_{speaker_folder}_{current_time}.wav"
        output_path = os.path.join(output_dir, output_file)
        
        ap.save_wav(wav, output_path)
        print(f"Saved synthesized audio to: {output_path}")
        
        # Return the audio data directly for playback
        return (ap.sample_rate, wav), output_path
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"Error in synthesis: {str(e)}"

def process_input_audio_async(audio_path, model=None, C=None, ap=None, speaker_manager=None):
    """A simpler version of process_input_audio that focuses on stability"""
    try:
        print("Starting audio processing...")
        
        # Create a timestamp-based unique ID for this processing session
        process_id = datetime.now().strftime('%Y%m%d%H%M%S')
        
        # Create a unique directory for this file to avoid resampling the entire folder
        process_dir = os.path.join("./processed_audio", f"proc_{process_id}")
        os.makedirs(process_dir, exist_ok=True)
        
        # Handle gradio audio input (tuple of sample_rate and audio_data)
        if isinstance(audio_path, tuple):
            print("Processing audio from recording...")
            sample_rate, audio_data = audio_path
            temp_path = os.path.join(process_dir, f"recording_{process_id}.wav")
            import scipy.io.wavfile as wav
            wav.write(temp_path, sample_rate, audio_data)
            audio_path = temp_path
        else:
            print(f"Processing audio from file: {audio_path}")
        
        if not os.path.exists(audio_path):
            return None, "Audio file not found"
        
        filename = os.path.basename(audio_path)
        basename = os.path.splitext(filename)[0]
        
        # Define processed path
        processed_wav_path = os.path.join(process_dir, f"{basename}_processed.wav")
        
        # Copy the file to process dir (simpler)
        try:
            shutil.copy(audio_path, processed_wav_path)
            print(f"Copied to processed path: {processed_wav_path}")
        except Exception as e:
            print(f"Error copying: {e}")
            return None, f"Error copying audio: {str(e)}"
        
        # Extract embedding if speaker_manager is available
        embedding = None
        if speaker_manager:
            try:
                print("Extracting speaker embedding...")
                embedding = speaker_manager.compute_embedding_from_clip(processed_wav_path)
                print("Embedding extraction complete")
            except Exception as e:
                print(f"Error extracting embedding: {e}")
                # Continue without embedding
        
        print(f"Audio processing completed successfully: {processed_wav_path}")
        return processed_wav_path, embedding, None
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, f"Error processing audio: {str(e)}"

def create_ui():
    """Create the Gradio interface with simplified UI"""
    # Get available model options
    model_options = prepare_model_dropdown_options()
    default_model = model_options[0][1] if model_options else None
    
    initial_status = "Please load a model to begin" if default_model else "No models found in the 'model' directory"
    
    # Create interface
    with gr.Blocks(title="Voice Dubbing System", theme=gr.themes.Soft()) as app:
        # Create state variables
        processed_audio_path_state = gr.State(None)
        speaker_embedding_state = gr.State(None)
        audio_processed_state = gr.State(False)
        model_state = gr.State(None)
        config_state = gr.State(None)
        ap_state = gr.State(None)
        speaker_manager_state = gr.State(None)
        model_initialized_state = gr.State(False)
        
        # Top header with model selection dropdown
        with gr.Row(equal_height=True):
            with gr.Column(scale=3):
                model_dropdown = gr.Dropdown(
                    choices=model_options,
                    value=default_model,
                    label="Model checkpoint",
                    interactive=True,
                    container=False,
                    scale=2
                )
            
            with gr.Column(scale=1, min_width=120):
                load_btn = gr.Button("Load Model", size="sm", variant="primary")
                refresh_btn = gr.Button("🔄 Refresh", size="sm")
            
            with gr.Column(scale=2):
                model_status = gr.Markdown(initial_status)
        
        gr.Markdown("---")
        
        # Main tabs
        with gr.Tabs() as tabs:
            # Dubbing tab (main functionality)
            with gr.TabItem("Dubbing"):
                # Audio input section
                gr.Markdown("## Input Reference Audio")
                with gr.Row():
                    audio_input = gr.Audio(
                        label="Record or Upload Audio", 
                        type="filepath"
                    )
                
                # Process and Transcribe buttons
                with gr.Row():
                    process_btn = gr.Button("Process Audio", variant="secondary")
                    transcribe_btn = gr.Button("Transcribe", variant="primary", interactive=False)
                
                # Processing status
                with gr.Row():
                    processing_status = gr.Markdown("Upload audio and click 'Process Audio' to start")
                
                # Transcription and translation outputs
                with gr.Row():
                    transcription_output = gr.Textbox(
                        label="Original Transcription",
                        placeholder="Transcription will appear here...",
                        lines=4
                    )
                
                with gr.Row():
                    translation_output = gr.Textbox(
                        label="Thai Translation",
                        placeholder="Translation will appear here...",
                        lines=4
                    )
                
                # Advanced Options (collapsible)
                with gr.Accordion("Advanced Options", open=False):
                    with gr.Row():
                        length_scale = gr.Slider(
                            minimum=0.5, 
                            maximum=2.5, 
                            value=1.5, 
                            step=0.1, 
                            label="Speech Length (Speed)",
                            info="Higher values make speech slower"
                        )
                    
                    with gr.Row():
                        noise_scale = gr.Slider(
                            minimum=0.0, 
                            maximum=1.0, 
                            value=0.2, 
                            step=0.05, 
                            label="Voice Variation",
                            info="Controls variation in voice characteristics"
                        )
                    
                    with gr.Row():
                        noise_scale_dp = gr.Slider(
                            minimum=0.0, 
                            maximum=1.0, 
                            value=0.2, 
                            step=0.05, 
                            label="Duration Variation",
                            info="Controls variation in phoneme durations"
                        )
                
                # Synthesis button
                with gr.Row():
                    synthesis_btn = gr.Button("Synthesize", variant="primary")
                
                # Status and output
                status_output = gr.Textbox(label="Status", interactive=False)
                
                # Output audio section
                gr.Markdown("## Output")
                with gr.Row():
                    audio_output = gr.Audio(label="Synthesized Audio")
                
                with gr.Row():
                    output_path_display = gr.Textbox(
                        label="Output File Path",
                        interactive=False
                    )
            
            # Placeholder tabs for future functionality
            with gr.TabItem("Zero-shot Inference (Work in Progress)"):
                gr.Markdown("## Coming Soon")
                gr.Markdown("This feature is under development.")
            
            with gr.TabItem("Training (Work in Progress)"):
                gr.Markdown("## Coming Soon")
                gr.Markdown("This feature is under development.")
        
        # Set up event handling
        
        # Load model button click
        def handle_load_model(checkpoint_name):
            if not checkpoint_name:
                return gr.Markdown("No checkpoint selected"), None, None, None, None, False
            
            try:
                # Initialize the model with selected checkpoint
                model, C, ap, speaker_manager, error = initialize_model(checkpoint_name)
                is_initialized = model is not None and C is not None and ap is not None and speaker_manager is not None
                
                if is_initialized:
                    # Get the model directory
                    checkpoint_info = scan_models()
                    model_dir = checkpoint_info.get(checkpoint_name, "unknown")
                    return gr.Markdown(f"✅ Model loaded: {model_dir}/{checkpoint_name}"), model, C, ap, speaker_manager, True
                else:
                    return gr.Markdown(f"❌ Failed to load checkpoint: {error}"), None, None, None, None, False
            except Exception as e:
                return gr.Markdown(f"❌ Error: {str(e)}"), None, None, None, None, False
        
        # Load model button click
        load_btn.click(
            fn=handle_load_model,
            inputs=model_dropdown,
            outputs=[
                model_status, 
                model_state, 
                config_state, 
                ap_state, 
                speaker_manager_state, 
                model_initialized_state
            ]
        )
        
        # Refresh button click
        def handle_refresh():
            new_options = prepare_model_dropdown_options()
            default = new_options[0][1] if new_options else None
            return gr.Dropdown(choices=new_options, value=default)
        
        refresh_btn.click(
            fn=handle_refresh,
            inputs=None,
            outputs=model_dropdown
        )
        
        # Handle audio processing in multiple steps to prevent UI freeze
        def start_processing():
            """First step: Show processing started message"""
            return "🔄 Processing audio... This may take a moment. Please wait."
        
        # Process audio step 2: Do the actual processing
        def process_audio(audio_input, model, C, ap, speaker_manager, is_model_initialized):
          """Process the audio and return explicit UI updates"""
          if not audio_input:
              return gr.update(value="Please provide an audio input (record or upload)."), gr.update(interactive=False), gr.update(value="⚠️ No audio provided"), None, None
          
          if not is_model_initialized:
              return gr.update(value="Model not initialized. Please load a model first."), gr.update(interactive=False), gr.update(value="⚠️ No model loaded"), None, None
          
          try:
              import time
              # Add a slight pause to ensure UI updates happen
              time.sleep(0.5)
              
              # Use the simplified processing function
              processed_path, embedding, error = process_input_audio_async(
                  audio_input, model, C, ap, speaker_manager)
              
              if error:
                  return gr.update(value=f"Error: {error}"), gr.update(interactive=False), gr.update(value=f"❌ Processing failed: {error}"), None, None
              
              print(f"Audio processed successfully: {processed_path}")
              
              # Return results with explicit gr.update()
              return gr.update(value="✅ Audio processing complete! You can now transcribe it."), gr.update(interactive=True), gr.update(value="Audio processed successfully"), processed_path, embedding
          
          except Exception as e:
              import traceback
              traceback.print_exc()
              return gr.update(value=f"Error: {str(e)}"), gr.update(interactive=False), gr.update(value=f"❌ Error: {str(e)}"), None, None
                
        # First show status message
        process_btn.click(
            fn=start_processing,
            inputs=None,
            outputs=processing_status,
            queue=False  # Important: Don't queue this to show feedback immediately
        )
        
        # Then do the actual processing as a separate event
        process_btn.click(
            fn=process_audio,
            inputs=[
                audio_input, 
                model_state, 
                config_state, 
                ap_state, 
                speaker_manager_state, 
                model_initialized_state
            ],
            outputs=[
                processing_status,
                transcribe_btn, 
                status_output, 
                processed_audio_path_state, 
                speaker_embedding_state
            ],
            queue=True,  # Queue this for background processing
            api_name="process_audio"
        )
        
        # Transcribe button click
        def handle_transcribe(audio_path, is_processed):
            if not audio_path:
                return None, None, "No processed audio found. Please process the audio first."
            
            # Get API key from environment
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                return None, None, "API Key not found. Please restart the application and enter your API key."
            
            try:
                transcription, translation, error = transcribe_audio(audio_path, api_key)
                if error:
                    return None, None, error
                return transcription, translation, "Transcription completed successfully."
            except Exception as e:
                import traceback
                traceback.print_exc()
                return None, None, f"Error: {str(e)}"
        
        transcribe_btn.click(
            fn=handle_transcribe,
            inputs=[processed_audio_path_state, audio_processed_state],
            outputs=[transcription_output, translation_output, status_output]
        )
        
        # Synthesis button click
        def handle_synthesis(audio_path, translation_text, model, C, ap, speaker_manager, embedding, 
                           model_initialized, model_option, length_scale, noise_scale, noise_scale_dp):
            # Check if a model is loaded
            if not model_initialized:
                # Try to load the model if not already loaded
                if model_option:
                    model, C, ap, speaker_manager, error = initialize_model(model_option)
                    model_initialized = model is not None and C is not None and ap is not None and speaker_manager is not None
                    if not model_initialized:
                        return None, None, f"Failed to load model: {error}"
                else:
                    return None, None, "No model selected. Please select a model from the dropdown."
            
            # Check if audio has been processed
            if not audio_path:
                return None, None, "Please process the audio first by clicking 'Process Audio'."
            
            # Use processed audio
            audio_to_use = audio_path
            
            if not translation_text or translation_text.strip() == "":
                return None, None, "Please transcribe the audio or enter translation text manually."
            
            try:
                # Use the advanced options and pass the pre-computed speaker embedding
                audio_result, output_file = synthesize_speech(
                    audio_to_use, 
                    translation_text, 
                    model, 
                    C, 
                    ap, 
                    speaker_manager,
                    speaker_embedding=embedding,
                    length_scale=length_scale,
                    noise_scale=noise_scale,
                    noise_scale_dp=noise_scale_dp
                )
                
                if not audio_result:
                    return None, None, f"Synthesis failed: {output_file}"
                
                return audio_result, output_file, "Synthesis completed successfully."
            except Exception as e:
                import traceback
                traceback.print_exc()
                return None, None, f"Error during synthesis: {str(e)}"
        
        synthesis_btn.click(
            fn=handle_synthesis,
            inputs=[
                processed_audio_path_state, 
                translation_output, 
                model_state, 
                config_state, 
                ap_state, 
                speaker_manager_state, 
                speaker_embedding_state, 
                model_initialized_state, 
                model_dropdown, 
                length_scale, 
                noise_scale, 
                noise_scale_dp
            ],
            outputs=[audio_output, output_path_display, status_output]
        )
    
    return app

def main():
    """Main function to run the webui"""
    # Create output directory
    os.makedirs(OUT_PATH, exist_ok=True)
    os.makedirs("./processed_audio", exist_ok=True)
    os.makedirs("./reference_voice", exist_ok=True)
    
    # Setup API key before launching the interface
    api_key = setup_api_key()
    
    # Create the UI
    ui = create_ui()
    
    # Enable queue for the entire application 
    ui.queue(concurrency_count=1, max_size=20)
    
    # Launch the interface
    ui.launch(
        server_name="0.0.0.0", 
        share=False,
        max_threads=1,
        show_error=True,
        prevent_thread_lock=True,
        debug=True
    )

if __name__ == "__main__":
    main()