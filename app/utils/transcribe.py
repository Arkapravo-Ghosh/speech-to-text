from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import Audio, Dataset
import librosa
import os


async def transcribe(audio_content_path):
    try:
        # Check if the CACHE_DIR environment variable is set
        cache_dir_exists = False
        cache_dir = os.getenv("CACHE_DIR")
    
        model_id = "openai/whisper-tiny.en"
        
        if cache_dir:
            cache_dir_exists = True

        # Load the processor
        if cache_dir_exists:
            processor = WhisperProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        else:
            processor = WhisperProcessor.from_pretrained(model_id)

        # Load the model
        if cache_dir_exists:
            model = WhisperForConditionalGeneration.from_pretrained(model_id, cache_dir=cache_dir)
        else:
            model = WhisperForConditionalGeneration.from_pretrained(model_id)

        # Move model to CPU
        model.to("cpu")

        # Model configuration
        forced_decoder_ids = processor.get_decoder_prompt_ids(
            language="en", task="transcribe"
        )

        model.config.forced_decoder_ids = forced_decoder_ids

        # Converting the audio file to a waveform
        wavform, sr = librosa.load(audio_content_path, sr=16000)

        # Creating a dataset from the waveform
        audio_dataset = Dataset.from_dict(
            {
                "audio": [
                    {
                        "array": wavform.tolist(),
                        "sampling_rate": sr,
                    }
                ]
            }
        ).cast_column("audio", Audio())

        sample = audio_dataset[0]["audio"]

        # Transcribing the audio
        input_features = processor(
            sample["array"],
            sampling_rate=sample["sampling_rate"],
            return_tensors="pt",
        ).input_features

        # Move input_features to CPU
        input_features = input_features.to("cpu")

        predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[
            0
        ]
        return transcription

    except Exception as error:
        print(f"Error during transcription: {str(error)}")
        return f"Server Error"
