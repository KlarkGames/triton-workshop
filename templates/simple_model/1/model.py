import torch
import triton_python_backend_utils as pb_utils
import torch


class TritonPythonModel:
    def initialize(self, args):
        print(args["model_config"])
        print(args["model_instance_kind"])
        print(args["model_instance_device_id"])
        print(args["model_repository"])
        print(args["model_version"])
        print(args["model_name"])
        print("INITIALIZATION COMPLETE")

    def execute(self, requests):
        responses = []

        for request in requests:
            a = pb_utils.get_input_tensor_by_name(request, "A").as_numpy()[0]
            b = pb_utils.get_input_tensor_by_name(request, "B").as_numpy()[0]
            
            result = torch.rand((a, b))
            print(f"Result's device: {result.device}", flush=True)

            response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor("RESULT", result.numpy()),
                ]
            )
            responses.append(response)

        return responses
    
    def finalize(self):
        print("I'LL BE BACK")

    