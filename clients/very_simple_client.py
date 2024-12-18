import click
import numpy as np
import grpc
from tritonclient.grpc import InferenceServerClient, InferInput, InferRequestedOutput

@click.command()
@click.option('--triton_address', help='Triton server address')
@click.option('--triton_port', type=int, help='Triton server port')
@click.option('--model_name', default="simple_model", show_default=True, help='Model name')
@click.option('--a', type=int, help='A value')
@click.option('--b', type=int, help='B value')
def client(triton_address, triton_port, model_name, a, b):
    try:
        triton_client = InferenceServerClient(url=f"{triton_address}:{triton_port}")

        a_input = InferInput("A", [1], "INT64")
        b_input = InferInput("B", [1], "INT64")

        a_input.set_data_from_numpy(np.asarray([a], dtype=np.int64))
        b_input.set_data_from_numpy(np.asarray([b], dtype=np.int64))
        
        output = InferRequestedOutput("RESULT")

        response = triton_client.infer(
            model_name=model_name,
            inputs=[a_input, b_input],
            outputs=[output]
        )

        output = response.as_numpy("RESULT")
        print(output.shape)
        

    except grpc.RpcError as e:
        print(f"GRPC error: {e.details()} (code: {e.code()})")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    client()
