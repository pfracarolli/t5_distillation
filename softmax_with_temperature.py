import matplotlib.pyplot as plt
import torch

n = 7
data = torch.tensor([0,.2,.4,.8,1,.8,.4,.2,0])
fig, ax = plt.subplots(1,n)
fig.set_figheight(2)

for i in range(0,n):
    t = (i + 1)

    ax[i].set_axis_off()
    ax[i].set_title(f'T={t}')
    ax[i].bar(torch.arange(len(data)), (data / t).softmax(dim=-1))

plt.show()