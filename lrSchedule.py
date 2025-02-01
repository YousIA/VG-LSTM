
def lr_schedule(epoch):
    if epoch < 100:
        return 0.001
    elif epoch < 200:
        return 0.0005
    else:
        return 0.0001
