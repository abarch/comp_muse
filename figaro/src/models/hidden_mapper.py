import torch.optim
import torch.nn as nn

from constants import PAD_TOKEN


class HiddenMapper(nn.Module):
    r"""
    This network is made to map an input of the dimension (#batches, #bar, 256) -> (#batches, #bar, 256)
    Since #bar may vary, and we do not want that
    """

    def __init__(self):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(512, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512)
        )

    def forward(self, x):
        return torch.stack([
            self.layer_stack(x_i) for x_i in torch.unbind(x, axis=0)
        ])


class ConditionalSeq2Seq(nn.Module):

    def __init__(self, figaro):
        super().__init__()
        self.figaro = figaro
        self.hidden_mapper = HiddenMapper()

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.figaro.vocab.to_i(PAD_TOKEN))



    def forward(self, x, z=None, labels=None, position_ids=None, bar_ids=None, description_bar_ids=None,
                return_hidden=False):
        encoder_hidden = self.figaro.encode(z, desc_bar_ids=description_bar_ids)

        encoder_hidden = self.hidden_mapper(encoder_hidden)

        out = self.figaro.decode(x,
                                 labels=labels,
                                 bar_ids=bar_ids,
                                 position_ids=position_ids,
                                 encoder_hidden_states=encoder_hidden,
                                 return_hidden=return_hidden
                                 )

        return out

    def get_loss(self, batch, return_logits=False):
        x = batch['input_ids']
        bar_ids = batch['bar_ids']
        position_ids = batch['position_ids']
        # Shape of labels: (batch_size, tgt_len, tuple_size)
        labels = batch['labels']

        # Shape of z: (batch_size, context_size, n_groups, d_latent)
        if self.figaro.description_flavor == 'latent':
            z = batch['latents']
            desc_bar_ids = None
        elif self.figaro.description_flavor == 'description':
            z = batch['description']
            desc_bar_ids = batch['desc_bar_ids']
        elif self.figaro.description_flavor == 'both':
            z = {'latents': batch['latents'], 'description': batch['description']}
            desc_bar_ids = batch['desc_bar_ids']
        else:
            z, desc_bar_ids = None, None

        logits = self(x, z=z, labels=labels, bar_ids=bar_ids, position_ids=position_ids,
                      description_bar_ids=desc_bar_ids)

        pred = logits.view(-1, logits.shape[-1])
        labels = labels.reshape(-1)

        loss = self.loss_fn(pred, labels)

        if return_logits:
            return loss, logits
        else:
            return loss

