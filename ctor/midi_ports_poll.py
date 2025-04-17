import mido
import json

ports = {
    "inputs": mido.get_input_names(),
    "outputs": mido.get_output_names()
}

print(json.dumps(ports))
