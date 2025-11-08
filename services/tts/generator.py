import os
import queue
import threading
import time
from typing import Generator as PyGenerator, Optional

import numpy as np
import torch
import torchaudio
import sounddevice as sd
import soundfile as sf

from transformers import CsmForConditionalGeneration, AutoProcessor
from transformers.models.csm.generation_csm import (
    GenerationConfig,
    LogitsProcessorList,
    StoppingCriteriaList,
)


def load_csm_1b(model_path="eustlb/csm-1b"):
    """
    Load CSM-1B model from path and return a ready-to-use Generator.

    Parameters
    ----------
    model_path : str
        Path to the model directory or HuggingFace model name

    Returns
    -------
    Generator
        A fully initialized generator instance ready to use
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CSM-1B model from path '{model_path}' to {device}...")
    model = CsmForConditionalGeneration.from_pretrained(model_path).to(device)
    processor = AutoProcessor.from_pretrained(model_path)
    model.generation_config.max_length = 250  # big enough to avoid recompilation
    model.generation_config.max_new_tokens = (
        None  # would take precedence over max_length
    )
    model.generation_config.cache_implementation = "static"
    model.depth_decoder.generation_config.cache_implementation = "static"
    print("Model loaded successfully!")

    # Create and return a generator instance
    generator = Generator(model, processor, device)
    run_warmup(generator)
    return generator


def run_warmup(generator):
    # Process input
    inputs = generator.processor.apply_chat_template(
        [
            {
                "role": "0",
                "content": [
                    {
                        "type": "text",
                        "text": "This is a warmup generation to make the model super fast.",
                    }
                ],
            }
        ],
        tokenize=True,
        return_dict=True,
    ).to(generator.device)

    # Stream generation with custom handling
    for i, chunk in enumerate(generator.generate_stream(inputs, chunk_token_size=20)):
        # chunk is a tensor containing audio samples
        # process or play each chunk as needed
        print(
            f"Generated warmup chunk {i+1} with {len(chunk.cpu().numpy()) / 24000:.3f} seconds of audio"
        )


class Generator:
    def __init__(self, model, processor, device):
        """
        Initialize a CSM generator with a model and processor.

        Parameters
        ----------
        model : CsmForConditionalGeneration
            The CSM model
        processor : AutoProcessor
            The CSM processor
        device : str
            The device to run the model on
        """
        self.model = model
        self.processor = processor
        self.device = device

        # Set default generation config
        self.model.generation_config.max_length = (
            250  # big enough to avoid recompilation
        )
        self.model.generation_config.max_new_tokens = (
            None  # would take precedence over max_length
        )
        self.model.generation_config.cache_implementation = "static"
        self.model.depth_decoder.generation_config.cache_implementation = "static"

    def _generate_stream(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_values: Optional[torch.Tensor] = None,
        input_values_cutoffs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        synced_gpus: Optional[bool] = None,
        *,
        chunk_token_size: int = 20,
        **kwargs,
    ) -> PyGenerator[torch.FloatTensor, None, None]:
        """
        Streams audio output from CSM model, yielding chunks as they are generated.

        Parameters
        ----------
        chunk_token_size:
            Number of codebook tokens to generate before yielding an audio chunk.
            With 24 kHz EnCodec, this equates to chunk_token_size * 20 ms of audio.

        Yields
        ------
        torch.FloatTensor
            Audio chunks as they are generated, in the range [-1, 1].
        """
        # Initialize RTF metrics tracking
        start_time_total = time.time()
        total_audio_duration = 0.0
        chunk_count = 0
        rtf_values = []
        sample_rate = 24000  # EnCodec sample rate
        first_chunk_generated = False
        first_chunk_start_time = start_time_total

        # Initialize generation config
        if generation_config is None:
            generation_config = self.model.generation_config

        # Initialize processors
        if logits_processor is None:
            logits_processor = LogitsProcessorList()

        if stopping_criteria is None:
            stopping_criteria = StoppingCriteriaList()

        # Ensure inputs are on the correct device
        device = self.device
        if input_ids is not None and input_ids.device != device:
            input_ids = input_ids.to(device)
        if input_values is not None and input_values.device != device:
            input_values = input_values.to(device)
        if input_values_cutoffs is not None and input_values_cutoffs.device != device:
            input_values_cutoffs = input_values_cutoffs.to(device)

        # Get convenience aliases
        eos_id = self.model.config.codebook_eos_token_id
        num_codebooks = self.model.config.num_codebooks

        # Initialize model kwargs
        model_kwargs = dict(kwargs)

        # Properly initialize cache_position
        cur_len = input_ids.shape[1] if input_ids.ndim > 1 else 1
        model_kwargs = self.model._get_initial_cache_position(
            cur_len, device, model_kwargs
        )

        # Generate initial tokens
        model_inputs = self.model.prepare_inputs_for_generation(
            input_ids,
            input_values=input_values,
            input_values_cutoffs=input_values_cutoffs,
            **model_kwargs,
        )

        # First forward pass to get initial tokens
        outputs = self.model(
            **model_inputs, return_dict=True, output_hidden_states=True
        )
        model_kwargs = self.model._update_model_kwargs_for_generation(
            outputs, model_kwargs
        )

        # Get next token prediction from initial output
        next_token_logits = outputs.logits[:, -1, :].float()

        # Process logits
        next_token_scores = logits_processor(input_ids, next_token_logits)

        # Sample or greedy selection
        if generation_config.do_sample:
            probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)

        # Get tokens for other codebooks
        first_codebook_ids = next_tokens[:, None]
        depth_decoder_input_ids = torch.nn.functional.pad(
            first_codebook_ids, (1, 0), value=0
        )
        backbone_last_hidden_state = outputs.hidden_states[-1][:, -1, :]

        # Generate tokens for other codebooks
        depth_decoder_outputs = self.model.depth_decoder.generate(
            input_ids=depth_decoder_input_ids,
            backbone_last_hidden_state=backbone_last_hidden_state.clone(),
        )

        # Process outputs
        if isinstance(depth_decoder_outputs, torch.Tensor):
            codebook_ids = depth_decoder_outputs[:, 1:]  # Remove placeholder
        else:
            codebook_ids = depth_decoder_outputs.sequences[:, 1:]

        # Initialize audio token collection
        audio_codebooks = [[] for _ in range(num_codebooks)]
        for i in range(min(num_codebooks, codebook_ids.size(1))):
            audio_codebooks[i].append(codebook_ids[0, i].item())
        token_count = 1

        # Main generation loop
        eos_generated = False
        while not eos_generated:
            # Format input for next iteration
            if input_ids.dim() == 2:  # [batch, seq_len]
                # This is the first iteration with text-only input
                # Reshape to [batch, seq_len=1, num_codebooks]
                if codebook_ids.size(1) < num_codebooks:
                    # Pad with EOS tokens if needed
                    padding = torch.full(
                        (1, num_codebooks - codebook_ids.size(1)),
                        eos_id,
                        dtype=codebook_ids.dtype,
                        device=device,
                    )
                    codebook_ids_padded = torch.cat([codebook_ids, padding], dim=1)
                    next_input_ids = codebook_ids_padded.unsqueeze(
                        1
                    )  # [batch, 1, num_codebooks]
                else:
                    next_input_ids = codebook_ids.unsqueeze(
                        1
                    )  # [batch, 1, num_codebooks]
            else:
                # For subsequent iterations
                if codebook_ids.size(1) < num_codebooks:
                    padding = torch.full(
                        (1, num_codebooks - codebook_ids.size(1)),
                        eos_id,
                        dtype=codebook_ids.dtype,
                        device=device,
                    )
                    codebook_ids_padded = torch.cat([codebook_ids, padding], dim=1)
                    next_codebook_ids = codebook_ids_padded.unsqueeze(
                        1
                    )  # [batch, 1, num_codebooks]
                else:
                    next_codebook_ids = codebook_ids.unsqueeze(
                        1
                    )  # [batch, 1, num_codebooks]

                # Concatenate with previous inputs
                next_input_ids = torch.cat([input_ids, next_codebook_ids], dim=1)

            # Update input_ids for next iteration
            input_ids = next_input_ids

            # Prepare inputs for next generation step
            model_inputs = self.model.prepare_inputs_for_generation(
                input_ids, **model_kwargs
            )
            model_inputs["output_hidden_states"] = True

            # Generate next token
            outputs = self.model(**model_inputs, return_dict=True)
            model_kwargs = self.model._update_model_kwargs_for_generation(
                outputs, model_kwargs
            )

            # Get next token logits
            next_token_logits = outputs.logits[:, -1, :].float()
            next_token_scores = logits_processor(input_ids, next_token_logits)

            # Sample or greedy selection for first codebook
            if generation_config.do_sample:
                probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            # Process with depth decoder for remaining codebooks
            first_codebook_ids = next_tokens[:, None]  # [batch, 1]
            depth_decoder_input_ids = torch.nn.functional.pad(
                first_codebook_ids, (1, 0), value=0
            )
            backbone_last_hidden_state = outputs.hidden_states[-1][:, -1, :]

            depth_decoder_outputs = self.model.depth_decoder.generate(
                input_ids=depth_decoder_input_ids,
                backbone_last_hidden_state=backbone_last_hidden_state.clone(),
            )

            # Process outputs
            if isinstance(depth_decoder_outputs, torch.Tensor):
                codebook_ids = depth_decoder_outputs[:, 1:]  # Remove placeholder
            else:
                codebook_ids = depth_decoder_outputs.sequences[:, 1:]

            # Check for EOS (all codebooks have EOS token)
            if (codebook_ids == eos_id).all():
                eos_generated = True

            # Add to audio tokens collection per codebook
            for i in range(min(num_codebooks, codebook_ids.size(1))):
                audio_codebooks[i].append(codebook_ids[0, i].item())
            token_count += 1

            # Check if we have enough tokens to yield audio
            if token_count >= chunk_token_size or eos_generated:
                # Start timing for this chunk
                chunk_start_time = time.time()

                # Convert collected tokens to the proper format for the codec model
                # The codec model expects [batch, codebooks, seq_len]
                max_len = max(len(tokens) for tokens in audio_codebooks)

                if max_len > 0:  # Only if we have tokens to process
                    audio_tensor = torch.full(
                        (1, num_codebooks, max_len),
                        eos_id,
                        dtype=torch.long,
                        device=device,
                    )

                    # Fill in the actual tokens
                    for i, tokens in enumerate(audio_codebooks):
                        if tokens:  # Only if we have tokens for this codebook
                            audio_tensor[0, i, : len(tokens)] = torch.tensor(
                                tokens, dtype=torch.long, device=device
                            )

                    # Decode audio from the tokens
                    with torch.no_grad():
                        decoded_audio = self.model.codec_model.decode(
                            audio_tensor
                        ).audio_values
                        chunk_audio = decoded_audio[0, 0]  # [samples]

                        # Calculate RTF for this chunk
                        chunk_end_time = time.time()
                        chunk_gen_time = chunk_end_time - chunk_start_time
                        audio_len_samples = chunk_audio.size(0)
                        audio_duration = audio_len_samples / sample_rate
                        chunk_rtf = (
                            chunk_gen_time / audio_duration if audio_duration > 0 else 0
                        )

                        # Update totals for average RTF
                        total_audio_duration += audio_duration
                        chunk_count += 1
                        rtf_values.append(chunk_rtf)

                        # Calculate running average RTF
                        avg_rtf = (
                            (chunk_end_time - start_time_total) / total_audio_duration
                            if total_audio_duration > 0
                            else 0
                        )

                        # Calculate first chunk latency (if this is the first chunk)
                        if not first_chunk_generated:
                            first_chunk_latency = (
                                chunk_end_time - first_chunk_start_time
                            )
                            print(f"First chunk latency: {first_chunk_latency:.3f}s")
                            first_chunk_generated = True

                        print(
                            f"Chunk {chunk_count}: RTF = {chunk_rtf:.4f} (chunk duration: {audio_duration:.3f}s, gen time: {chunk_gen_time:.3f}s)"
                        )
                        print(f"Running average RTF: {avg_rtf:.4f}")

                        yield chunk_audio

                    # Reset token collection except for the last token (for continuity)
                    last_tokens = [
                        tokens[-1] if tokens else eos_id for tokens in audio_codebooks
                    ]
                    audio_codebooks = [[] for _ in range(num_codebooks)]

                    # Don't add the last tokens if we've reached EOS
                    if not eos_generated:
                        for i, token in enumerate(last_tokens):
                            audio_codebooks[i].append(token)
                        token_count = 1

            # Clean up to prevent memory issues
            del outputs
            del depth_decoder_outputs

        # Final summary stats
        if chunk_count > 0:
            total_time = time.time() - start_time_total
            final_rtf = (
                total_time / total_audio_duration if total_audio_duration > 0 else 0
            )

            print(f"FINAL RTF METRICS:")
            print(f"  Total generation time: {total_time:.3f}s")
            print(f"  Total audio duration: {total_audio_duration:.3f}s")
            print(f"  Overall RTF: {final_rtf:.4f}")
            if rtf_values:
                print(f"  Min chunk RTF: {min(rtf_values):.4f}")
                print(f"  Max chunk RTF: {max(rtf_values):.4f}")
                print(f"  Avg chunk RTF: {sum(rtf_values)/len(rtf_values):.4f}")

    def generate_stream(self, inputs, chunk_token_size=20, **kwargs):
        """
        Stream audio chunks as they're generated.

        Parameters
        ----------
        inputs : dict
            Processed inputs from the processor
        chunk_token_size : int, optional
            Number of tokens to generate before yielding an audio chunk, by default 20
        **kwargs : dict
            Additional arguments to pass to the generator

        Yields
        ------
        torch.FloatTensor
            Audio chunks as they are generated
        """

        # Call the streaming generation method
        for chunk in self._generate_stream(
            **inputs, chunk_token_size=chunk_token_size, **kwargs
        ):
            yield chunk


def generate_streaming_audio(
    generator,
    conversation,
    output_filename=None,
    play_audio=True,
    chunk_token_size=20,
    reference_data=None,
    **kwargs,
):
    """
    Generate and play audio from a conversation, streaming chunks as they are generated.

    Parameters
    ----------
    generator : Generator
        The generator to use
    conversation : list or str
        Conversation history in the format expected by the processor, or a text prompt
    output_filename : str, optional
        Filename to save the generated audio to, by default None
    play_audio : bool, optional
        Whether to play the audio in real time, by default True
    chunk_token_size : int, optional
        Number of tokens to generate before yielding an audio chunk, by default 20
    reference_data : list, optional
        Reference audio data to include in the conversation, by default None
    **kwargs : dict
        Additional arguments to pass to the generator

    Returns
    -------
    numpy.ndarray
        The complete generated audio
    """
    # Set up audio queue and threading if playing audio
    audio_queue = queue.Queue()
    complete_audio = []

    if play_audio:

        def audio_playback_thread():
            while True:
                chunk = audio_queue.get()
                if chunk is None:  # None is our signal to stop
                    break
                sd.play(chunk.astype("float32"), 24000)
                sd.wait()  # Wait here is fine as it's in a separate thread
                audio_queue.task_done()

        # Start playback thread
        playback_thread = threading.Thread(target=audio_playback_thread)
        playback_thread.daemon = True
        playback_thread.start()

    try:
        # Process the conversation
        final_conversation = []

        # Add reference data if provided
        if reference_data:
            print(
                f"Adding {len(reference_data)} reference audio sample(s) to conversation"
            )
            for ref in reference_data:
                speaker_id = ref.get("speaker_id", "0")
                text = ref.get("text", "")
                audio_array = ref.get("audio_array", None)

                # Verify audio is properly formatted
                if (
                    audio_array is None
                    or not isinstance(audio_array, np.ndarray)
                    or audio_array.size == 0
                ):
                    print(f"Warning: Invalid reference audio array, using dummy audio")
                    audio_array = np.zeros(24000)  # 1 second of silence

                # If audio is a PyTorch tensor, convert to numpy
                if hasattr(audio_array, "numpy"):
                    audio_array = audio_array.numpy()

                print(
                    f"Adding reference audio: speaker={speaker_id}, text='{text[:30]}...', "
                    f"audio_shape={audio_array.shape}, audio_range=[{audio_array.min():.2f}, {audio_array.max():.2f}]"
                )

                # Create content entry with both text and audio
                content = [
                    {"type": "text", "text": text},
                    {"type": "audio", "audio": audio_array},
                ]

                final_conversation.append({"role": speaker_id, "content": content})

        # Add conversation data
        if isinstance(conversation, list):
            # Assume it's already in the right format
            final_conversation.extend(conversation)
        else:
            # Assume it's just text
            final_conversation.append(
                {
                    "role": "0",  # Default speaker ID
                    "content": [{"type": "text", "text": conversation}],
                }
            )

        # Process inputs with voice cloning information if provided
        print(
            f"Applying chat template to conversation with {len(final_conversation)} entries"
        )
        try:
            inputs = generator.processor.apply_chat_template(
                final_conversation, tokenize=True, return_dict=True
            ).to(generator.device)

            # Print token count to aid in debugging
            print(
                f"Generated {inputs['input_ids'].shape[1]} tokens for the conversation"
            )

        except Exception as e:
            print(f"Error applying chat template: {e}")
            import traceback

            traceback.print_exc()
            raise

        # Open WAV file for writing if output_filename provided
        if output_filename:
            wav_fh = sf.SoundFile(
                output_filename, "w", samplerate=24000, channels=1, subtype="PCM_16"
            )
        else:
            wav_fh = None

        print(f"Starting streaming generation...")

        # Stream generation and queue chunks as they arrive
        for pcm_chunk in generator.generate_stream(
            inputs, chunk_token_size=chunk_token_size, **kwargs
        ):
            # Convert to numpy and ensure correct dtype
            np_chunk = pcm_chunk.cpu().numpy().astype(np.float32)
            complete_audio.append(np_chunk)

            # Add to queue if playing
            if play_audio:
                audio_queue.put(np_chunk)

            # Save to file if provided
            if wav_fh:
                wav_fh.write(np_chunk)

            # Print progress indicator
            print(".", end="", flush=True)

        # Close the WAV file if opened
        if wav_fh:
            wav_fh.close()
            print(f"\nSaved to '{output_filename}'")

        # Wait for playback to finish if playing
        if play_audio:
            audio_queue.join()
            audio_queue.put(None)  # Signal to stop the thread
            playback_thread.join()

        # Concatenate all chunks
        if complete_audio:
            full_audio = np.concatenate(complete_audio)
            print(f"Generated {len(full_audio) / 24000:.2f} seconds of audio")
            return full_audio
        else:
            print("No audio was generated")
            return np.array([])

    except Exception as e:
        print(f"Error during generation: {e}")
        import traceback

        traceback.print_exc()

        # Clean up
        if play_audio and "playback_thread" in locals() and playback_thread.is_alive():
            audio_queue.put(None)

        return None


def load_reference_audio(reference_data):
    """
    Load and process reference audio for the CSM model.

    Parameters
    ----------
    reference_data : list
        List of dictionaries containing reference data:
        - path: Path to the audio file
        - text: Text corresponding to the audio
        - speaker_id: Speaker ID for the audio

    Returns
    -------
    list
        Processed reference data with audio arrays
    """
    processed_data = []
    for ref in reference_data:
        ref_copy = ref.copy()  # Create a copy to avoid modifying the original

        try:
            # Load audio file
            waveform, sample_rate = torchaudio.load(ref_copy["path"])

            # Resample to 24000 Hz if needed (CSM-1B operates at this rate)
            if sample_rate != 24000:
                resampler = torchaudio.transforms.Resample(sample_rate, 24000)
                waveform = resampler(waveform)

            # Convert to mono if needed
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Normalize audio to be in the range [-1, 1]
            if waveform.abs().max() > 0:
                waveform = waveform / waveform.abs().max()

            # Convert tensor to numpy array
            ref_copy["audio_array"] = waveform.squeeze().cpu().numpy()
            ref_copy["sample_rate"] = 24000
            print(f"Loaded reference audio: {os.path.basename(ref_copy['path'])}")

            # Verify audio is properly loaded
            audio_duration = len(ref_copy["audio_array"]) / 24000
            print(f"Reference audio duration: {audio_duration:.2f} seconds")
            print(
                f"Audio range: min={ref_copy['audio_array'].min():.4f}, max={ref_copy['audio_array'].max():.4f}"
            )

        except Exception as e:
            print(f"Warning: Could not load reference audio {ref_copy['path']}: {e}")
            # Provide dummy audio as fallback
            ref_copy["audio_array"] = np.zeros(24000)  # 1 second of silence
            ref_copy["sample_rate"] = 24000

        processed_data.append(ref_copy)

    return processed_data
