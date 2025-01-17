import numpy as np
import matplotlib.pyplot as plt

# with open("out/metrics/ForwardDiffusion_metrics_log.txt") as fp:
#     lines = fp.readlines()
#     lines = [l.strip().replace(' ', '').split('|') for l in lines]
#     losses = [float(l[1].split('-')[1]) for l in lines if l[0] != '']
#     plt.plot(losses)
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.show()


with open("out/metrics/ForwardDiffusion_VB+MSE_CosineBeta_Fine1_metrics_log.txt") as fp:
    lines = fp.readlines()
    lines = [l.strip().replace(' ', '').split('|') for l in lines]
    losses = [float(l[1].split('-')[1]) for l in lines if l[0] != '']
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()