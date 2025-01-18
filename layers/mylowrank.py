import torch
import torch.nn as nn
import math


class LowRank(nn.Module):
    def __init__(self, in_features, out_features, rank, bias=True):
        super(LowRank, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank= rank
        self.bias = bias

        wA = torch.empty(self.in_features, rank)
        wB = torch.empty(self.rank, self.out_features)

        self.A = nn.Parameter(nn.init.kaiming_uniform_(wA))
        self.B = nn.Parameter(nn.init.kaiming_uniform_(wB))

        if self.bias:
            wb = torch.empty(self.out_features)
            self.b = nn.Parameter(nn.init.uniform_(wb))

    def forward(self, x):
        out = x @ self.A
        out = out @ self.B

        if self.bias:
            out += self.b

        return out








#######################################

class ReducedLinear(nn.Module):
    def __init__(self, in_features, out_features, rank, bias=True, tol=1e-2):
        super(ReducedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank= min(rank, in_features, out_features)
        self.rmax = rank
        self.bias = bias
        self.tol = tol

        wU = torch.empty(in_features, self.rmax)
        nn.init.orthogonal_(wU)
        self.U = nn.Parameter(wU)

        wV = torch.empty(out_features, self.rmax)
        nn.init.orthogonal_(wV)
        self.V = nn.Parameter(wV)

        wS = torch.abs(torch.empty(self.rmax))
        nn.init.uniform_(wS, a=0.1, b=1.0)  # Initialize with positive values
        self.S = nn.Parameter(torch.diag(wS))

        if self.bias:
          self.b = nn.Parameter(torch.zeros(out_features))
        else:
          self.b = None

    def forward(self, x):

        # Apply the low-rank linear transformation
        out = x @ self.U[:, :self.rank]
        out = out @ self.S[:self.rank, :self.rank]
        out = out @ self.V[:, :self.rank].T

        # Add bias if applicable
        if self.bias:
            out = out + self.b
        return out


    @torch.no_grad()
    def step(self):
        # Re-orthogonalize U and V using QR decomposition
        self.U.data, R1 = torch.linalg.qr(self.U.data)
        self.V.data, R2 = torch.linalg.qr(self.V.data)     
        self.S.data = R1 @ self.S.data @ R2.T


#############################################



class ThinFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, U, S, V, bias=None):
        # Save input and weight for backward pass
        ctx.save_for_backward(input, U, S, V, bias)
        output = input @ U
        output = output @ S
        output = output @ V.T
        if bias is not None:
            output += bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        input, U, S, V, bias = ctx.saved_tensors

        # Compute gradients for input, weight, and bias
        grad_input = grad_U = grad_S = grad_V = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output @ V
            grad_input = grad_input @ S
            grad_input = grad_input @ U

        if ctx.needs_input_grad[1]:
            grad_U = input.transpose(1,2) @ grad_output
            grad_U = grad_U @ V
            grad_U = grad_U @ S
            grad_U = torch.mean(grad_U, dim=0, keepdim=False)

            grad_U = (grad_U @ U.T - U @ grad_U.T) @ U



        if ctx.needs_input_grad[2]:
            grad_S = input @ U
            grad_S = grad_S.transpose(1,2) @ grad_output
            grad_S = grad_S @ V
            grad_S = torch.mean(grad_S, dim=0, keepdim=False)
            grad_S = torch.diag(torch.diagonal(grad_S))



        if ctx.needs_input_grad[3]:
            grad_V = input @ U
            grad_V = grad_V @ S
            grad_V = grad_V.transpose(1,2) @ grad_output
            grad_V = grad_V.transpose(1,2)
            grad_V = torch.mean(grad_V, dim=0, keepdim=False)

            grad_V = (grad_V @ V.T - V @ grad_V.T) @ V

        if bias is not None and ctx.needs_input_grad[4]:
            grad_bias = torch.mean(grad_output, dim=[1, 0])

        return grad_input, grad_U, grad_S, grad_V, grad_bias


class ThinLinear(nn.Module):
    def __init__(self, in_features, out_features, rank, bias=True):
        super(ThinLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.bias = bias

        wU = torch.empty(in_features, rank)
        nn.init.orthogonal_(wU)
        self.U = nn.Parameter(wU)

        wV = torch.empty(out_features, rank)
        nn.init.orthogonal_(wV)
        self.V = nn.Parameter(wV)

        wS = torch.empty(rank)
        nn.init.uniform_(wS, a=0.01, b=0.1)
        self.S = nn.Parameter(torch.diag(wS))

        if self.bias:
          self.b = nn.Parameter(torch.zeros(out_features))
        else:
          self.b = None

        
    def forward(self, x):
        return ThinFunction.apply(x, self.U, self.S, self.V, self.b)
    
    def step(self):

        pass
        # with torch.no_grad():
        #     self.U.data, R1 = torch.linalg.qr(self.U.data)
        #     self.V.data, R2 = torch.linalg.qr(self.V.data)
        #     self.S.data = R1 @ self.S.data @ R2.T


