import click
import librosa
import numpy as np
from scipy.io.wavfile import write
import grpc
from tritonclient.grpc import InferenceServerClient, InferInput, InferRequestedOutput

@click.command()
@click.option('--triton-address', help='Triton server address')
@click.option('--triton-port', type=int, help='Triton server port')
@click.option('--model-name', default="enhancer_ensemble", show_default=True, help='Model name')
@click.option('--input-file', help='Input file path')
@click.option('--chunk-duration', type=float, default=30.0, show_default=True, help='Chunk duration')
@click.option('--chunk-overlap', type=float, default=1.0, show_default=True, help='Chunk overlap')
@click.option('--save-path', type=str, default="output.wav", show_default=True, help='Save path')
def client(triton_address, triton_port, model_name, input_file, chunk_duration, chunk_overlap, save_path):
    try:
        triton_client = InferenceServerClient(url=f"{triton_address}:{triton_port}")

        audio_data, sample_rate = librosa.load(input_file, sr=None)

        audio_input = InferInput("INPUT_AUDIO", [len(audio_data)], "FP32")
        sample_rate_input = InferInput("SAMPLE_RATE", [1], "INT64")
        chunk_duration_input = InferInput("CHUNK_DURATION_S", [1], "FP32")
        chunk_overlap_input = InferInput("CHUNK_OVERLAP_S", [1], "FP32")

        audio_input.set_data_from_numpy(audio_data.astype(np.float32))
        sample_rate_input.set_data_from_numpy(np.asarray([sample_rate], dtype=np.int64))
        chunk_duration_input.set_data_from_numpy(np.asarray([chunk_duration], dtype=np.float32))
        chunk_overlap_input.set_data_from_numpy(np.asarray([chunk_overlap], dtype=np.float32))

        output = InferRequestedOutput("OUTPUT_AUDIO")

        response = triton_client.infer(
            model_name=model_name,
            inputs=[audio_input, sample_rate_input, chunk_duration_input, chunk_overlap_input],
            outputs=[output]
        )

        output_audio = response.as_numpy("OUTPUT_AUDIO")

        write(save_path, 44100, output_audio.astype(np.float32))

    except grpc.RpcError as e:
        print(f"GRPC error: {e.details()} (code: {e.code()})")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    client()
