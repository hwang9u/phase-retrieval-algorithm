
import librosa
import numpy as np
import pandas as pd
import librosa.display
import matplotlib.pyplot as plt

# Reference
# T. Peer, S. Welker, and T. Gerkmann, “Beyond Griffin-Lim: Improved Iterative Phase Retrieval for Speech,” in Int. Workshop on Acoustic Signal Enhancement (IWAENC), Sep. 2022.


### Griffin Lim Algorithm(GLA) ###
def P1(X, A):
    """
    주어진 복소 행렬 X의 크기를 주어진 A로 바꿔줌

    Args:
        X (array): 복소 행렬
        A (array): 대체할 크기 행렬

    Returns:
        array: X' = |A| exp( j _X) 
    """
    return A * np.exp(1j * np.angle(X))

def P2(X, istft_params = {} , stft_params = None):
    if stft_params == None:
        stft_params =   istft_params.copy()
        stft_params['n_fft'] = (X.shape[0]-1)*2
    x_rec = librosa.istft(X, **istft_params)
    _X = librosa.stft(x_rec, **stft_params)
    return _X

def GriffinLim(A, n_iters = 32, istft_params = {}):
    P = np.random.uniform( size = np.prod(A.shape),
                          low = -np.pi,
                          high = np.pi
                                    ).reshape(*A.shape)
    S = A * np.exp(1j * P)
    for _ in range(n_iters-1):
        S = P2(X=P1(S, A), stft_params=None,istft_params= istft_params)
    
    y = librosa.istft( S, **istft_params)
    return y

### Relaxed Averaged Alternating Reflections(RAAR) ###
def RAAR(A,  istft_params = {},n_iters = 32, beta = .9):
    P = np.random.uniform( size = np.prod(A.shape),
                          low = -np.pi,
                          high = np.pi
                                    ).reshape(*A.shape)
    S = A * np.exp(1j * P)
    for k in range(n_iters-1):
        Ra = 2* P1(S, A) - S
        Rc = 2 * P2(Ra, istft_params= istft_params) - Ra
        S = 0.5 * beta * (S + Rc) + (1 - beta) * P1(S, A)
    
    y = librosa.istft( S, **istft_params)
    return y


### Difference Map(DM) ###
def fA(X,A, b = 0.9):
    return P1(X, A) + (P1(X, A) - X)/b

def fC(X, istft_params, b = 0.9):
    return P2(X,
              istft_params=istft_params) - \
    (P2(X, istft_params=istft_params) -X)/b

def DM(A,istft_params = {}, n_iters = 10, beta = 0.7):
    P = np.random.uniform( size = np.prod(A.shape),
                          low = -np.pi,
                          high = np.pi
                                    ).reshape(*A.shape)
    S = A * np.exp(1j * P)
    for _ in range(n_iters-1):
        S = S + beta * (P2( fA(S, A, b = beta), istft_params=istft_params) 
                        - P1(fC(S, istft_params=istft_params, b = beta) ,A) )
    y = librosa.istft(S, **istft_params)
    return y




