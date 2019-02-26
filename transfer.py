from torch import nn

def transfer(model, n_classes):
    """
    Run transfer learning
    """

    model_name = model.__class__.__name__
    # Freeze early layers
    for param in model.parameters():
        param.requires_grad = False

    if model_name == 'LeNet':
        n_inputs = model.last_layers[1].in_features
        model.last_layers = nn.Sequential(
            nn.Linear(n_inputs, 128, bias=True),
            nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Linear(128, n_classes, bias=True)
        )
    elif model_name == 'PerturbResNet':
        n_inputs = model.linear.in_features
        model.linear = nn.Linear(n_inputs, n_classes, bias=True)
    else:
        raise ValueError(f"Unknown model {model_name}")

    model = model.to('cuda')
    return model

