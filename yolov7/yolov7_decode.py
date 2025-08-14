import torch



def decode (output,stride,anchor,grid,nc,bs):
    
    z = []
    
    for i in range(len(anchor)):
        y = output[i].sigmoid()
        xy, wh, conf = y.split((2, 2, nc + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
        xy = xy * (2. * stride[i]) + (stride[i] * (grid[i] - 0.5))  # new xy
        wh = wh ** 2 * (4 * anchor[i].data)  # new wh
        y = torch.cat((xy, wh, conf), 4)
        z.append(y.view(bs, -1, nc+5))


    return (torch.cat(z, 1), output)