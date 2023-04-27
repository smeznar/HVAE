import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

from tree import Node, BatchedNode
from symbol_library import SymType


class HVAE(nn.Module):
    _symbols = None

    def __init__(self, input_size, output_size, hidden_size=None):
        super(HVAE, self).__init__()

        if hidden_size is None:
            hidden_size = output_size

        self.encoder = Encoder(input_size, hidden_size, output_size)
        self.decoder = Decoder(output_size, hidden_size, input_size)

    def forward(self, tree):
        mu, logvar = self.encoder(tree)
        z = self.sample(mu, logvar)
        out = self.decoder(z, tree)
        return mu, logvar, out

    def sample(self, mu, logvar):
        eps = Variable(torch.randn(mu.size()))
        std = torch.exp(logvar / 2.0)
        return mu + eps * std

    def encode(self, tree):
        mu, logvar = self.encoder(tree)
        return mu, logvar

    def decode(self, z):
        if HVAE._symbols is None:
            raise Exception("To generate expression trees, a symbol library is needed. Please add it using the"
                            " HVAE.add_symbols method.")
        return self.decoder.decode(z, HVAE._symbols)

    @staticmethod
    def add_symbols(symbols):
        HVAE._symbols = symbols
        Node.add_symbols(symbols)
        BatchedNode.add_symbols(symbols)


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.gru = GRU221(input_size=input_size, hidden_size=hidden_size)
        self.mu = nn.Linear(in_features=hidden_size, out_features=output_size)
        self.logvar = nn.Linear(in_features=hidden_size, out_features=output_size)

        torch.nn.init.xavier_uniform_(self.mu.weight)
        torch.nn.init.xavier_uniform_(self.logvar.weight)

    def forward(self, tree):
        # Check if the tree has target vectors
        if tree.target is None:
            tree.add_target_vectors()

        tree_encoding = self.recursive_forward(tree)
        mu = self.mu(tree_encoding)
        logvar = self.logvar(tree_encoding)
        return mu, logvar

    def recursive_forward(self, tree):
        left = self.recursive_forward(tree.left) if tree.left is not None \
            else torch.zeros(tree.target.size(0), 1, self.hidden_size)
        right = self.recursive_forward(tree.right) if tree.right is not None \
            else torch.zeros(tree.target.size(0), 1, self.hidden_size)

        # left = left.mul(tree.mask[:, None, None])
        # right = right.mul(tree.mask[:, None, None])

        hidden = self.gru(tree.target, left, right)
        hidden = hidden.mul(tree.mask[:, None, None])
        return hidden


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.z2h = nn.Linear(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.gru = GRU122(input_size=output_size, hidden_size=hidden_size)

        torch.nn.init.xavier_uniform_(self.z2h.weight)
        torch.nn.init.xavier_uniform_(self.h2o.weight)

    # Used during training to guide the learning process
    def forward(self, z, tree):
        hidden = self.z2h(z)
        self.recursive_forward(hidden, tree)
        return tree

    def recursive_forward(self, hidden, tree):
        prediction = self.h2o(hidden)
        symbol_probs = F.softmax(prediction, dim=2)
        tree.prediction = prediction
        if tree.left is not None or tree.right is not None:
            left, right = self.gru(symbol_probs, hidden)
            if tree.left is not None:
                self.recursive_forward(left, tree.left)
            if tree.right is not None:
                self.recursive_forward(right, tree.right)

    # Used for inference to generate expression trees from latent vectorS
    def decode(self, z, symbol_dict):
        with torch.no_grad():
            mask = torch.ones(z.size(0)).bool()
            hidden = self.z2h(z)
            batch = self.recursive_decode(hidden, symbol_dict, mask)
            return batch.to_expr_list()

    def recursive_decode(self, hidden, symbol_dict, mask):
        prediction = F.softmax(self.h2o(hidden), dim=2)
        # Sample symbol in a given node
        symbols, left_mask, right_mask = self.sample_symbol(prediction, symbol_dict, mask)
        left, right = self.gru(prediction, hidden)
        if torch.any(left_mask):
            l_tree = self.recursive_decode(left, symbol_dict, left_mask)
        else:
            l_tree = None

        if torch.any(right_mask):
            r_tree = self.recursive_decode(right, symbol_dict, right_mask)
        else:
            r_tree = None

        node = BatchedNode()
        node.symbols = symbols
        node.left = l_tree
        node.right = r_tree
        return node

    def sample_symbol(self, prediction, symbol_dict, mask):
        sampled = F.softmax(prediction, dim=2)
        # Select the symbol with the highest value ("probability")
        symbols = []
        left_mask = torch.clone(mask)
        right_mask = torch.clone(mask)

        for i in range(sampled.size(0)):
            if mask[i]:
                symbol = symbol_dict[torch.argmax(sampled[i, 0, :])]
                symbols.append(symbol["symbol"])
                if symbol["type"].value is SymType.Fun.value:
                    right_mask[i] = False
                if symbol["type"].value is SymType.Var.value or symbol["type"].value is SymType.Const.value:
                    left_mask[i] = False
                    right_mask[i] = False
            else:
                symbols.append("")
        return symbols, left_mask, right_mask


class GRU221(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRU221, self).__init__()
        self.wir = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.whr = nn.Linear(in_features=2*hidden_size, out_features=hidden_size)
        self.wiz = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.whz = nn.Linear(in_features=2 * hidden_size, out_features=hidden_size)
        self.win = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.whn = nn.Linear(in_features=2 * hidden_size, out_features=hidden_size)
        torch.nn.init.xavier_uniform_(self.wir.weight)
        torch.nn.init.xavier_uniform_(self.whr.weight)
        torch.nn.init.xavier_uniform_(self.wiz.weight)
        torch.nn.init.xavier_uniform_(self.whz.weight)
        torch.nn.init.xavier_uniform_(self.win.weight)
        torch.nn.init.xavier_uniform_(self.whn.weight)

    def forward(self, x, h1, h2):
        h = torch.cat([h1, h2], dim=2)
        r = torch.sigmoid(self.wir(x) + self.whr(h))
        z = torch.sigmoid(self.wiz(x) + self.whz(h))
        n = torch.tanh(self.win(x) + r * self.whn(h))
        return (1 - z) * n + (z / 2) * h1 + (z / 2) * h2


class GRU122(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRU122, self).__init__()
        self.hidden_size = hidden_size
        self.wir = nn.Linear(in_features=input_size, out_features=2*hidden_size)
        self.whr = nn.Linear(in_features=hidden_size, out_features=2*hidden_size)
        self.wiz = nn.Linear(in_features=input_size, out_features=2*hidden_size)
        self.whz = nn.Linear(in_features=hidden_size, out_features=2*hidden_size)
        self.win = nn.Linear(in_features=input_size, out_features=2*hidden_size)
        self.whn = nn.Linear(in_features=hidden_size, out_features=2*hidden_size)
        torch.nn.init.xavier_uniform_(self.wir.weight)
        torch.nn.init.xavier_uniform_(self.whr.weight)
        torch.nn.init.xavier_uniform_(self.wiz.weight)
        torch.nn.init.xavier_uniform_(self.whz.weight)
        torch.nn.init.xavier_uniform_(self.win.weight)
        torch.nn.init.xavier_uniform_(self.whn.weight)

    def forward(self, x, h):
        r = torch.sigmoid(self.wir(x) + self.whr(h))
        z = torch.sigmoid(self.wiz(x) + self.whz(h))
        n = torch.tanh(self.win(x) + r * self.whn(h))
        dh = h.repeat(1, 1, 2)
        out = (1 - z) * n + z * dh
        return torch.split(out, self.hidden_size, dim=2)
