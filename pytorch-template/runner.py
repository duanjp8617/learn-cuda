import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "build"))
import vector_add  # noqa: E402

def main():
    input1 = torch.randn(1000, device="cuda")
    input2 = torch.randn(1000, device="cuda")
    output = vector_add.add(input1, input2)
    torch.testing.assert_close(output, input1 + input2)

    print("done")


if __name__ == "__main__":
    main()
