import click
from pytriton.client import ModelClient
import librosa
import numpy as np
from scipy.io.wavfile import write

@click.command()
@click.option('--triton-address', help='Triton server address')
@click.option('--triton-port', type=int, help='Triton server port')
@click.option('--model-name', default="enhancer_ensemble", show_default=True, help='Model name')
@click.option('--input-file', help='Input file path')
@click.option('--chunk-duration', type=float, default=30.0, show_default=True, help='Chunk duration')
@click.option('--chunk-overlap', type=float, default=1.0, show_default=True, help='Chunk overlap')
@click.option('--save-path', type=str, default="output.wav", show_default=True, help='Save path')
def client(triton_address, triton_port, model_name, input_file, chunk_duration, chunk_overlap, save_path):
    client = ModelClient(f"{triton_address}:{triton_port}", model_name)

    audio_data, sample_rate = librosa.load(input_file)

    output = client.infer_sample(
        INPUT_AUDIO=audio_data,
        SAMPLE_RATE=np.asarray([sample_rate]),
        CHUNK_DURATION_S=np.asarray([chunk_duration], dtype=np.float32),
        CHUNK_OVERLAP_S=np.asarray([chunk_overlap], dtype=np.float32)
    )

    write(save_path, 44100, output["OUTPUT_AUDIO"])

if __name__ == "__main__":
    client()