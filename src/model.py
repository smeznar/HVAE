import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

from tree import Node
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
        if HVAE.symbols is None:
            raise Exception("To generate expression trees, a symbol library is needed. Please add it using the"
                            " HVAE.add_symbols method.")
        return self.decoder.decode(z, HVAE.symbols)

    @staticmethod
    def add_symbols(symbols):
        HVAE.symbols = symbols
        Node.add_symbols(symbols)


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
        hidden = self.gru(tree.target, left, right)
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

    # Used for inference to generate expression trees from latent vectors
    def decode(self, z, symbols):
        hidden = self.z2h(z)
        tree = self.recursive_decode(hidden, symbols)
        return tree

    def recursive_decode(self, hidden, symbols):
        prediction = self.h2o(hidden)
        # Sample symbol in a given node
        sampled, symbol, stype = Decoder.sample_symbol(prediction, symbols)
        if stype is SymType.Fun:
            left, right = self.gru(sampled, hidden)
            l_tree = self.recursive_decode(left, symbols)
            r_tree = None
        elif stype is SymType.Operator:
            left, right = self.gru(sampled, hidden)
            l_tree = self.recursive_decode(left, symbols)
            r_tree = self.recursive_decode(right, symbols)
        else:
            l_tree = None
            r_tree = None
        return Node(symbol, right=r_tree, left=l_tree)

    @staticmethod
    def sample_symbol(prediction, symbol_dict):
        sampled = F.softmax(prediction, dim=2)
        # Select the symbol with the highest value ("probability")
        symbol = symbol_dict[torch.argmax(sampled).item()]
        return sampled, symbol["symbol"], symbol["type"]


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
