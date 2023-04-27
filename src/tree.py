import torch
from torch.autograd import Variable


class Node:
    _symbols = None
    _s2c = None

    def __init__(self, symbol=None, right=None, left=None):
        self.symbol = symbol
        self.right = right
        self.left = left
        self.target = None
        self.prediction = None

    def __str__(self):
        return "".join(self.to_list())

    def __len__(self):
        left = len(self.left) if self.left is not None else 0
        right = len(self.right) if self.right is not None else 0
        return 1 + left + right

    def height(self):
        hl = self.left.height() if self.left is not None else 0
        hr = self.right.height() if self.right is not None else 0
        return max(hl, hr) + 1

    def to_list(self, notation="infix"):
        if notation == "infix" and Node._symbols is None:
            raise Exception("To generate a list of token in the infix notation, symbol library is needed. Please use"
                            " the Node.add_symbols methods to add them, before using the to_list method.")
        left = [] if self.left is None else self.left.to_list(notation)
        right = [] if self.right is None else self.right.to_list(notation)
        if notation == "prefix":
            return [self.symbol] + left + right
        elif notation == "postfix":
            return left + right + [self.symbol]
        elif notation == "infix":
            if len(left) > 0 and len(right) == 0 and Node.symbol_precedence(self.symbol) > 0:
                return [self.symbol] + ["("] + left + [")"]
            elif len(left) > 0 >= Node.symbol_precedence(self.symbol) and len(right) == 0:
                return ["("] + left + [")"] + [self.symbol]

            if self.left is not None \
                    and -1 < Node.symbol_precedence(self.left.symbol) < Node.symbol_precedence(self.symbol):
                left = ["("] + left + [")"]
            if self.right is not None \
                    and -1 < Node.symbol_precedence(self.right.symbol) < Node.symbol_precedence(self.symbol):
                right = ["("] + right + [")"]
            return left + [self.symbol] + right
        else:
            raise Exception("Invalid notation selected. Use 'infix', 'prefix', 'postfix'.")

    def to_pexpr(self):
        if Node._symbols is None:
            raise Exception("To generate a pexpr, symbol library is needed. Please use"
                            " the Node.add_symbols methods to add them, before using the to_list method.")
        left = [] if self.left is None else self.left.to_pexpr()
        right = [] if self.right is None else self.right.to_pexpr()
        return [Node._symbols[Node._s2c[self.symbol]]["psymbol"]] + left + right

    def add_target_vectors(self):
        if Node._symbols is None:
            raise Exception("Encoding needs a symbol library to create target vectors. Please use Node.add_symbols"
                            " method to add a symbol library to trees before encoding.")
        target = torch.zeros(len(Node._symbols)).float()
        target[Node._s2c[self.symbol]] = 1.0
        self.target = Variable(target[None, None, :])
        if self.left is not None:
            self.left.add_target_vectors()
        if self.right is not None:
            self.right.add_target_vectors()

    def loss(self, mu, logvar, lmbda, criterion):
        pred = Node.to_matrix(self, "prediction")
        target = Node.to_matrix(self, "target")
        BCE = criterion(pred, target)
        KLD = (lmbda * -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))
        return BCE + KLD, BCE, KLD

    def clear_prediction(self):
        if self.left is not None:
            self.left.clear_prediction()
        if self.right is not None:
            self.right.clear_prediction()
        self.prediction = None

    def to_dict(self):
        d = {'s': self.symbol}
        if self.left is not None:
            d['l'] = self.left.to_dict()
        if self.right is not None:
            d['r'] = self.right.to_dict()
        return d

    @staticmethod
    def from_dict(d):
        left = None
        right = None
        if "l" in d:
            left = Node.from_dict(d["l"])
        if 'r' in d:
            right = Node.from_dict(d["r"])
        return Node(d["s"], right=right, left=left)

    @staticmethod
    def symbol_precedence(symbol):
        return Node._symbols[Node._s2c[symbol]]["precedence"]

    @staticmethod
    def to_matrix(tree, matrix_type="prediction"):
        reps = []
        if tree.left is not None:
            reps.append(Node.to_matrix(tree.left, matrix_type))

        if matrix_type == "target":
            reps.append(torch.Tensor([Node._s2c[tree.symbol]]).long())
        else:
            reps.append(tree.prediction[0, :, :])

        if tree.right is not None:
            reps.append(Node.to_matrix(tree.right, matrix_type))

        return torch.cat(reps)

    @staticmethod
    def add_symbols(symbols):
        Node._symbols = symbols
        Node._s2c = {s["symbol"]: i for i, s in enumerate(symbols)}


class BatchedNode():
    _symbols = None
    _s2c = None

    def __init__(self, size=0, trees=None):
        self.symbols = ["" for _ in range(size)]
        self.left = None
        self.right = None

        if trees is not None:
            for tree in trees:
                self.add_tree(tree)

    @staticmethod
    def add_symbols(symbols):
        BatchedNode._symbols = symbols
        BatchedNode._s2c = {s["symbol"]: i for i, s in enumerate(symbols)}

    def add_tree(self, tree=None):
        if tree is None:
            self.symbols.append("")

            if self.left is not None:
                self.left.add_tree()
            if self.right is not None:
                self.right.add_tree()
        else:
            self.symbols.append(tree.symbol)

            if self.left is not None and tree.left is not None:
                self.left.add_tree(tree.left)
            elif self.left is not None:
                self.left.add_tree()
            elif tree.left is not None:
                self.left = BatchedNode(size=len(self.symbols)-1)
                self.left.add_tree(tree.left)

            if self.right is not None and tree.right is not None:
                self.right.add_tree(tree.right)
            elif self.right is not None:
                self.right.add_tree()
            elif tree.right is not None:
                self.right = BatchedNode(size=len(self.symbols)-1)
                self.right.add_tree(tree.right)

    def loss(self, mu, logvar, lmbda, criterion):
        pred = BatchedNode.get_prediction(self)
        pred = torch.permute(pred, [0, 2, 1])
        target = BatchedNode.get_target(self)
        BCE = criterion(pred, target)
        KLD = (lmbda * -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))/mu.size(0)
        return BCE + KLD, BCE, KLD

    def create_target(self):
        if BatchedNode._symbols is None:
            raise Exception("Encoding needs a symbol library to create target vectors. Please use"
                            " BatchedNode.add_symbols method to add a symbol library to trees before encoding.")
        target = torch.zeros((len(self.symbols), 1, len(Node._symbols)))
        mask = torch.ones(len(self.symbols))

        for i, s in enumerate(self.symbols):
            if s == "":
                mask[i] = 0
            else:
                target[i, 0, Node._s2c[s]] = 1

        self.mask = mask
        self.target = Variable(target)

        if self.left is not None:
            self.left.create_target()
        if self.right is not None:
            self.right.create_target()

    def to_expr_list(self):
        exprs = []
        for i in range(len(self.symbols)):
            exprs.append(self.get_expr_at_idx(i))
        return exprs

    def get_expr_at_idx(self, idx):
        symbol = self.symbols[idx]
        if symbol == "":
            return None

        left = self.left.get_expr_at_idx(idx) if self.left is not None else None
        right = self.right.get_expr_at_idx(idx) if self.right is not None else None

        return Node(symbol, left=left, right=right)

    @staticmethod
    def get_prediction(tree):
        reps = []
        if tree.left is not None:
            reps.append(BatchedNode.get_prediction(tree.left))

        target = tree.prediction[:, 0, :]
        reps.append(target[:, None, :])

        if tree.right is not None:
            reps.append(BatchedNode.get_prediction(tree.right))

        return torch.cat(reps, dim=1)

    @staticmethod
    def get_target(tree):
        reps = []
        if tree.left is not None:
            reps.append(BatchedNode.get_target(tree.left))

        target = torch.zeros(len(tree.symbols)).long()
        for i, s in enumerate(tree.symbols):
            if s == "":
                target[i] = -1
            else:
                target[i] = BatchedNode._s2c[s]
        reps.append(target[:, None])

        if tree.right is not None:
            reps.append(BatchedNode.get_target(tree.right))

        return torch.cat(reps, dim=1)



