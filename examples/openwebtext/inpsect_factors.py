import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from tueplots import markers

from kronfluence.analyzer import Analyzer


def main():
    plt.rcParams.update({"figure.dpi": 150})
    plt.rcParams.update(markers.with_edge())
    plt.rcParams["axes.axisbelow"] = True

    layer_num = 31
    module_name = f"model.layers.{layer_num}.mlp.down_proj"
    # module_name = f"model.layers.{layer_num}.mlp.up_proj"
    lambda_processed = Analyzer.load_file("num_lambda_processed.safetensors")[module_name]
    lambda_matrix = Analyzer.load_file("lambda_matrix.safetensors")[module_name]
    lambda_matrix.div_(lambda_processed)
    lambda_matrix = lambda_matrix.float()
    plt.matshow(lambda_matrix, cmap="PuBu", norm=LogNorm())

    plt.title(module_name)
    plt.colorbar()
    plt.show()
    plt.clf()

    lambda_matrix = lambda_matrix.view(-1).numpy()
    sorted_lambda_matrix = np.sort(lambda_matrix)
    plt.plot(sorted_lambda_matrix)
    plt.title(module_name)
    plt.grid()
    plt.yscale("log")
    plt.ylabel("Eigenvalues")
    plt.show()


if __name__ == "__main__":
    main()
