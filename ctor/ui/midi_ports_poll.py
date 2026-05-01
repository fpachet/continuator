import json

import mido


def main():
    ports = {
        "inputs": mido.get_input_names(),
        "outputs": mido.get_output_names()
    }
    print(json.dumps(ports))


if __name__ == "__main__":
    main()
