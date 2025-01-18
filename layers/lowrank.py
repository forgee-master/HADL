import torch
import torch.nn as nn


class LowRankLayer(nn.Module):
    def __init__(self, input_size, output_size, rank, adaptive=True, init_compression=0.5):
        """Constructs a low-rank layer of the form U*S*V'*x + b, where 
           U, S, V represent the facorized weight W and b is the bias vector
        Args:
            input_size: input dimension of weight W
            output_size: output dimension of weight W, dimension of bias b
            rank: initial rank of factorized weight
            adaptive: set True if layer isrank-adaptive
            init_compression: initial compression of neural network
        """
        # construct parent class nn.Module
        super(LowRankLayer, self).__init__()

        self.rmax = rank
        rmax = self.rmax
        if adaptive:
            r1 = 2*rmax
        else:  
            r1 = rank

        # initializes factorized weight
        self.U = nn.Parameter(torch.randn(input_size, r1))
        self.S = nn.Parameter(torch.randn(r1, r1))
        self.V = nn.Parameter(torch.randn(output_size, r1))

        # ensure that U and V are orthonormal
        self.U.data, _ = torch.linalg.qr(self.U, 'reduced')
        self.V.data, _ = torch.linalg.qr(self.V, 'reduced')

        # initialize non-trainable Parameter fields for S-step
        self.U1 = nn.Parameter(torch.randn(input_size, 2*rank))
        self.V1 = nn.Parameter(torch.randn(output_size, 2*rank))
        self.U1.requires_grad = False
        self.V1.requires_grad = False

        # initialize bias
        self.bias = nn.Parameter(torch.randn(output_size))

        # set rank and truncation tolerance
        self.r = int(init_compression * rank)
        self.tol = 1e-2
        self.adaptive = adaptive

    def forward(self, x):
        """Returns the output of the layer. The formula implemented is output = U*S*V'*x + bias.
        Args:
            x: input to layer
        Returns: 
            output of layer
        """
        r = self.r
        xU = torch.matmul(x, self.U[:,:r])
        xUS = torch.matmul(xU, self.S[:r,:r])
        out = torch.matmul(xUS, self.V[:,:r].T)
        return out + self.bias

    @torch.no_grad()
    def step(self, learning_rate, dlrt_step="basis"):
        """Performs a steepest descend training update on specified low-rank factors
        Args:
            learning_rate: learning rate for training
            dlrt_step: sepcifies step that is taken. Can be 'K', 'L' or 'S'
            adaptive: specifies if fixed-rank or rank-adaptivity is used
        """
        r = self.r
        if self.adaptive:
            r1 = 2*r
        else:
            r1 = r

        if dlrt_step == "basis":
            # perform K-step
            U0 = self.U[:,:r]
            V0 = self.V[:,:r]
            S0 = self.S[:r,:r]
            K = torch.matmul(U0, S0)
            dK = torch.matmul(self.U.grad[:,:r], S0)# + torch.matmul(U0, self.S.grad[:r,:r])
            if self.adaptive:
                self.U1[:,:r1], _ = torch.linalg.qr(torch.cat((U0, dK),1), 'reduced')
            else:
                K = K - learning_rate * dK
                self.U1[:,:r1], _ = torch.linalg.qr(K, 'reduced')

            # perform L-step
            L = torch.matmul(V0, S0.T)
            dL = torch.matmul(self.V.grad[:,:r], S0.T)# + torch.matmul(V0, self.S.grad[:r,:r].T)
            if self.adaptive:
                self.V1[:,:r1], _ = torch.linalg.qr(torch.cat((V0, dL),1), 'reduced')
            else:
                L = L - learning_rate * dL
                self.V1[:,:r1], _ = torch.linalg.qr(L, 'reduced')
                
            M = self.U1[:,:r1].T @ U0[:,:r]
            N = V0[:,:r].T @ self.V1[:,:r1]

            # update basis
            self.U.data[:,:r1] = self.U1[:,:r1]
            self.V.data[:,:r1] = self.V1[:,:r1]

            self.S.data[:r1,:r1] = M @ self.S[:r,:r] @ N

            self.r = r1

            # update bias
            self.bias.data = self.bias - learning_rate * self.bias.grad

        elif dlrt_step == "coefficients":
            self.S.data[:self.r,:self.r] = self.S[:self.r,:self.r] - learning_rate * self.S.grad[:self.r,:self.r]

            # truncate to new rank
            if self.adaptive:
                self.Truncate()
        else:
            print("Wrong step defined: ", dlrt_step)

    @torch.no_grad()
    def BiasStep(self, learning_rate):
        """Performs a steepest descend training update on the bias
        Args:
            learning_rate: learning rate for training
        """
        self.bias.data = self.bias - learning_rate * self.bias.grad

    @torch.no_grad()
    def Truncate(self):
        """Truncates the weight matrix to a new rank"""
        r0 = int(0.5*self.r)
        P, d, Q = torch.linalg.svd(self.S[:self.r, :self.r])

        #print(torch.linalg.matrix_norm(P @ torch.diag(d) @ Q.t() - self.S[:self.r, :self.r], 'fro'))

        tol = self.tol# * torch.linalg.norm(d)
        r1 = self.r
        for j in range(0, self.r):
            tmp = torch.linalg.norm(d[j:self.r])
            if tmp < tol:
                r1 = j
                break

        r1 = min(r1, self.rmax)
        r1 = max(r1, 2)
        r1 = r0

        # update s
        self.S.data[:r1, :r1] = torch.diag(d[:r1])

        # update u and v
        self.U.data[:, :r1] = torch.matmul(self.U[:, :self.r], P[:, :r1])
        self.V.data[:, :r1] = torch.matmul(self.V[:, :self.r], Q.T[:, :r1]) # DOUBLE CECK Q.T here. Should it be Q?
        self.r = int(r1)


# Define low-rank layer
class ParallelLowRankLayer(nn.Module):
    def __init__(self, input_size, output_size, rank, init_compression=0.5):
        """Constructs a low-rank layer of the form U*S*V'*x + b, where 
           U, S, V represent the facorized weight W and b is the bias vector
        Args:
            input_size: input dimension of weight W
            output_size: output dimension of weight W, dimension of bias b
            rank: initial rank of factorized weight
            init_compression: initial compression of neural network
        """
        # construct parent class nn.Module
        super(ParallelLowRankLayer, self).__init__()

        self.rmax = rank
        rmax = self.rmax
        r1 = 2*self.rmax

        # initializes factorized weight
        self.U = nn.Parameter(torch.randn(input_size, r1))
        self.V = nn.Parameter(torch.randn(output_size, r1))
        self.SK = nn.Parameter(torch.randn(rmax, rmax))
        self.SL = nn.Parameter(torch.randn(rmax, rmax))
        self.Sbar = nn.Parameter(torch.randn(rmax, rmax))
        self.SK.requires_grad = False
        self.SL.requires_grad = False
        self.Sbar.requires_grad = False

        # ensure that U and V are orthonormal
        self.U.data, _ = torch.linalg.qr(self.U, 'reduced')
        self.V.data, _ = torch.linalg.qr(self.V, 'reduced')

        # initialize non-trainable Parameter fields for S-step
        self.U1 = nn.Parameter(torch.randn(input_size, r1))
        self.V1 = nn.Parameter(torch.randn(output_size, r1))
        self.U1.requires_grad = False
        self.V1.requires_grad = False

        self.singular_values, _ = torch.sort(torch.randn(r1) ** 2, descending=True)
        self.S = nn.Parameter(torch.diag(self.singular_values))  # nn.Parameter(torch.eye(r1, r1))
        self.Sinv = torch.Tensor(torch.diag(1 / self.singular_values))  # , requires_grad=False)

        # initialize bias
        self.bias = nn.Parameter(torch.randn(output_size))

        # set rank and truncation tolerance
        self.r = self.r = int(init_compression * rank)
        self.tol = 1e-2

    def forward(self, x):
        """Returns the output of the layer. The formula implemented is output = U*S*V'*x + bias.
        Args:
            x: input to layer
        Returns: 
            output of layer
        """
        r = self.r
        xU = torch.matmul(x, self.U[:,:r])
        xUS = torch.matmul(xU, self.S[:r,:r])
        out = torch.matmul(xUS, self.V[:,:r].T)
        return out + self.bias

    @torch.no_grad()
    def step(self, learning_rate):
        """Performs a steepest descend training update on specified low-rank factors
        Args:
            learning_rate: learning rate for training
        """
        r = self.r
        r1 = 2*r

        U0 = self.U[:,:r]
        V0 = self.V[:,:r]
        S0 = self.S[:r,:r]
        Sinv = self.Sinv[:r,:r]

        # perform K-step
        K = torch.matmul(U0, S0)
        dK = torch.matmul(self.U.grad[:,:r], Sinv)
        K = K - learning_rate * dK
        self.U1[:,:r1], _ = torch.linalg.qr(torch.cat((U0, dK),1), 'reduced')
        UTilde = self.U1[:,r:r1]
        
        #print(torch.matmul(UTilde.T, K).shape)
        try:
            self.SK.data[:r,:r] = torch.matmul(UTilde.T, K)
        except: 
            print("K failed")
            print(r)
            print(UTilde.T.shape)
            print( K.shape)

        # perform L-step
        L = torch.matmul(V0, S0.T)
        dL = torch.matmul(self.V.grad[:,:r], Sinv)
        L = L - learning_rate * dL
        self.V1[:,:r1], _ = torch.linalg.qr(torch.cat((V0, dL),1), 'reduced')
        VTilde = self.V1[:,r:r1]
        try:
            self.SL.data[:r,:r] = torch.matmul(L.T, VTilde)
        except: 
            print("L failed")
            print(r)
            print(L.T.shape)
            print( VTilde.shape)

        # perform S-step
        self.Sbar.data[:r,:r] = self.S[:r,:r] - learning_rate * self.S.grad[:r,:r]

        # set up augmented S matrix
        self.S.data[:r,:r] = self.Sbar[:r,:r]
        self.S.data[r:r1,:r] = self.SK[:r,:r]
        self.S.data[:r,r:r1] = self.SL[:r,:r]
        self.S.data[r:r1,r:r1] = torch.zeros((r,r))

        # update basis
        self.U.data[:,:r1] = torch.cat((U0, UTilde),1)
        self.V.data[:,:r1] = torch.cat((V0, VTilde),1)
        self.r = r1

        # update bias
        self.bias.data = self.bias - learning_rate * self.bias.grad

        self.Truncate()


    @torch.no_grad()
    def BiasStep(self, learning_rate):
        """Performs a steepest descend training update on the bias
        Args:
            learning_rate: learning rate for training
        """
        self.bias.data = self.bias - learning_rate * self.bias.grad

    @torch.no_grad()
    def Truncate(self):
        """Truncates the weight matrix to a new rank"""
        r0 = int(0.5*self.r)
        P, d, Q = torch.linalg.svd(self.S[:self.r, :self.r])

        #print(torch.linalg.matrix_norm(P @ torch.diag(d) @ Q.t() - self.S[:self.r, :self.r], 'fro'))

        tol = self.tol * torch.linalg.norm(d)
        r1 = self.r
        for j in range(0, self.r):
            tmp = torch.linalg.norm(d[j:self.r])
            if tmp < tol:
                r1 = j
                break

        r1 = min(r1, self.rmax)
        r1 = max(r1, 2)

        # update s
        self.S.data[:r1, :r1] = torch.diag(d[:r1])
        self.Sinv[:r1, :r1] = torch.diag(1./d[:r1])

        # update u and v
        self.U.data[:, :r1] = torch.matmul(self.U[:, :self.r], P[:, :r1])
        self.V.data[:, :r1] = torch.matmul(self.V[:, :self.r], Q.T[:, :r1]) # DOUBLE CECK Q.T here. Should it be Q?
        self.r = int(r1)        