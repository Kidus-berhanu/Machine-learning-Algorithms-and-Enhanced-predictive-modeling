class VisionTransformer(nn.Module):
    def __init__(self, num_classes):
        super(VisionTransformer, self).__init__()
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=512, nhead=8), num_layers=6)
        self.fc = nn.Linear(512, num_classes)
        self.attention = nn.MultiheadAttention(d_model=512, nhead=8)

    def forward(self, x):
        x, _ = self.attention(x, x, x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x

    def init_weights(self):
        """Initialize the weights of the model."""
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def set_trainable(self, trainable=True):
        """Set the model to be trainable or not."""
        for param in self.parameters():
            param.requiresGrad = trainable

    def fine_tune(self, freeze_encoder=False):
        """Fine-tune the model.

        Args:
            freeze_encoder (bool): If True, the encoder layers will not be trained.
        """
        for name, param in self.named_parameters():
            if freeze_encoder and 'transformer' in name:
                param.requiresGrad = False
            else:
                param.requiresGrad = True

    def load_state_dict(self, state_dict, strict=True):
        """Load the state dict of the model.

        Args:
            state_dict (dict): The state dict of the model.
            strict (bool): Whether to strictly enforce that the keys in
                state_dict match the keys returned by this module's state_dict()
                function. Default: True
        """
        model_state = self.state_dict()
        model_state.update(state_dict)
        super(VisionTransformer, self).load_state_dict(model_state, strict)
