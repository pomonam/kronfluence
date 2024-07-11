import logging

import matplotlib.pyplot as plt

from kronfluence.analyzer import Analyzer


def main():
    logging.basicConfig(level=logging.INFO)

    name = "ekfac"
    factor = Analyzer.load_file(f"influence_results/cifar10/factors_{name}/activation_covariance.safetensors")

    plt.matshow(factor["6.0"])
    plt.show()

    factor = Analyzer.load_file(f"influence_results/cifar10/factors_{name}/gradient_covariance.safetensors")

    plt.matshow(factor["6.0"])
    plt.show()


if __name__ == "__main__":
    main()
