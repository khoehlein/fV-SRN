from matplotlib import pyplot as plt

from compression.experiments.plot_compression_stats import draw_compressor_stats
from volnet.experiments.paper.plot_data.plot_singleton_vs_ensemble import add_layout


def main():
    fig, axs = plt.subplots(2,3, figsize=(8, 4), sharex='all', sharey='row')
    draw_compressor_stats(axs, ['level'], 'reverted')
    add_layout(axs)
    axs[1, 1].legend()
    plt.tight_layout()
    # plt.savefig('classical_compression_performance.pdf')
    plt.show()


if __name__ == '__main__':
    main()
