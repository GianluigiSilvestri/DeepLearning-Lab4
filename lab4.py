import numpy as np
import matplotlib.pyplot as plt

class RNN():
    def __init__(self,m,K):
        self.b=np.zeros((m,1)) # bias vector, dimension (m,1), m is hidden state dimensionality
        self.c=np.zeros((K,1)) # bias vector, dimension (K,1), K is alphabet size

        #random initialization of weight matrices
        self.U=np.random.normal(0,0.1,(m,K))
        self.W=np.random.normal(0,0.1,(m,m))
        self.V=np.random.normal(0,0.1,(K,m))

    def initial_momentum(self):
        m_V=np.zeros((self.V.shape))
        m_W = np.zeros((self.W.shape))
        m_U = np.zeros((self.U.shape))
        m_c = np.zeros((self.c.shape))
        m_b = np.zeros((self.b.shape))

        return [m_V,m_W,m_U,m_c,m_b]

    def rnn_to_list(self):
        weights = [self.V, self.W, self.U, self.c, self.b]
        return weights

    def update(self,weights):
        self.V=weights[0]
        self.W=weights[1]
        self.U = weights[2]
        self.c = weights[3]
        self.b = weights[4]

INPUT_FILE='goblet_book.txt'

def read_data(file_path):

    with open(file_path,"r") as input:
        data=input.read()

    chars=list(set(data))
    chars.sort()
    indices=np.arange(len(chars))
    char_and_ind=np.asarray(list(zip(chars,indices)))
    return char_and_ind,data

def generate_chars(chars_with_indices,alph_size,first_char,rnn,n,h_0):
    labels=[]
    x_0 = to_onehot(alph_size, chars_with_indices, first_char)
    a_t=np.dot(rnn.W,h_0)+np.dot(rnn.U,x_0)+rnn.b
    h_t=np.tanh(a_t)
    o_t=np.dot(rnn.V,h_t)+rnn.c
    p_t=softmax(o_t)
    labels.append(char_from_index(chars_with_indices, np.where(np.random.multinomial(1, p_t[:, 0]) == 1)[0][0]))
    for t in range(n-1):
        x_t=to_onehot(alph_size,chars_with_indices,labels[-1])
        a_t = np.dot(rnn.W, h_t) + np.dot(rnn.U, x_t) + rnn.b
        h_t = np.tanh(a_t)
        o_t = np.dot(rnn.V, h_t) + rnn.c
        p_t = softmax(o_t)
        labels.append(char_from_index(chars_with_indices,np.where(np.random.multinomial(1, p_t[:, 0]) == 1)[0][0]))

    return ''.join(labels)


def softmax(S):
    return np.exp(S)/np.sum(np.exp(S),axis=0)

def index_from_char(chars_with_indices,wanted_char):
    index=np.where(chars_with_indices==wanted_char)[0][0]
    return index

def char_from_index(chars_with_indices,wanted_ind):
    char=chars_with_indices[wanted_ind][0]
    return char

def to_onehot(alph_size,chars_with_indices,my_char):
    x_0 = np.zeros((alph_size, 1), dtype=int)
    x_0[index_from_char(chars_with_indices, my_char)] += 1
    return x_0

def seq_to_ohm(alph_size,sequence,chars_with_indices):
    matrix=np.zeros((alph_size,len(sequence)),dtype=int)
    for c,char in enumerate(sequence):
        matrix[index_from_char(chars_with_indices, char ),c]+=1
    return matrix

def forward(X,rnn,h_prev):
    h_0=h_prev
    h = np.zeros((rnn.b.shape[0], np.shape(X)[1]))
    a = np.zeros((h.shape))
    probabilities=np.zeros((X.shape))
    for i,x in enumerate(X.T):
        if i==0:
            a[:,i] = (np.dot(rnn.W, h_0) + np.dot(rnn.U, x.reshape(-1,1)) + rnn.b)[:,0]
        else:
            a[:,i] = (np.dot(rnn.W, h[:,i-1].reshape(-1,1)) + np.dot(rnn.U, x.reshape(-1, 1)) + rnn.b)[:,0]
        h[:,i] = np.tanh(a[:,i])
        o_t = np.dot(rnn.V, h[:,i].reshape(-1,1)) + rnn.c
        p_t = softmax(o_t)
        probabilities[:,i]=p_t[:,0]

    return probabilities,h,a

def compute_loss(X,rnn,Y,h_prev):
    P,_,_=forward(X,rnn,h_prev)
    return -np.sum(np.log(np.sum(Y * P, axis=0)))

def check_grad(grad_a, grad_n, eps):
    '''function to compare the analitical (grad_a) and numerical (grad_n) gradients'''
    diff = np.abs(grad_a - grad_n) / max(eps, np.amax(np.abs(grad_a) + np.abs(grad_n)))
    if np.amax(diff) < 1e-6:
        return True
    else:
        return False

def compute_grad_num_slow(X, Y, rnn,h,h_prev):
    '''centered difference gradient for W and Fs'''

    V=rnn.V
    W=rnn.W
    U=rnn.U
    c=rnn.c
    b=rnn.b
    grad_V = np.zeros((np.shape(V)))
    grad_W= np.zeros((np.shape(W)))
    grad_U = np.zeros((np.shape(U)))
    grad_c=np.zeros((np.shape(c)))
    grad_b = np.zeros((np.shape(b)))

    it = np.nditer(V, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        iV = it.multi_index
        old_value = V[iV]
        V[iV] = old_value - h  # use original value
        c1 = compute_loss(X,rnn,Y,h_prev)
        V[iV] = old_value + h  # use original value
        c2 = compute_loss(X, rnn, Y,h_prev)
        grad_V[iV] = (c2 - c1) / (2 * h)
        V[iV] = old_value  # restore original value
        it.iternext()

    it = np.nditer(W, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        iW = it.multi_index
        old_value = W[iW]
        W[iW] = old_value - h  # use original value
        c1 = compute_loss(X,rnn,Y,h_prev)
        W[iW] = old_value + h  # use original value
        c2 = compute_loss(X,rnn,Y,h_prev)
        grad_W[iW] = (c2 - c1) / (2 * h)
        W[iW] = old_value  # restore original value
        it.iternext()

    it = np.nditer(U, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        iU = it.multi_index
        old_value = U[iU]
        U[iU] = old_value - h  # use original value
        c1 = compute_loss(X, rnn, Y,h_prev)
        U[iU] = old_value + h  # use original value
        c2 = compute_loss(X, rnn, Y,h_prev)
        grad_U[iU] = (c2 - c1) / (2 * h)
        U[iU] = old_value  # restore original value
        it.iternext()

    it = np.nditer(c, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ic = it.multi_index
        old_value = c[ic]
        c[ic] = old_value - h  # use original value
        c1 = compute_loss(X, rnn, Y,h_prev)
        c[ic] = old_value + h  # use original value
        c2 = compute_loss(X, rnn, Y,h_prev)
        grad_c[ic] = (c2 - c1) / (2 * h)
        c[ic] = old_value  # restore original value
        it.iternext()

    it = np.nditer(b, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ib = it.multi_index
        old_value = b[ib]
        b[ib] = old_value - h  # use original value
        c1 = compute_loss(X, rnn, Y,h_prev)
        b[ib] = old_value + h  # use original value
        c2 = compute_loss(X, rnn, Y,h_prev)
        grad_b[ib] = (c2 - c1) / (2 * h)
        b[ib] = old_value  # restore original value
        it.iternext()

    return [grad_V,grad_W, grad_U, grad_c, grad_b]

def compute_gradients(X,Y,rnn,h_prev):
    P,H,A=forward(X,rnn,h_prev)
    m=np.shape(H)[0] #100
    seq_lenght=np.shape(H)[1] #25
    G=-(Y-P).T
    next_h=H[:,-1].reshape(-1,1)

    grad_c=np.sum(G,axis=0).reshape(-1,1)
    grad_V=np.dot(G.T,H.T)
    grad_a=np.zeros((m,seq_lenght))
    grad_h=np.zeros((m,seq_lenght))
    grad_h[:,-1]=np.dot(G[-1,:],rnn.V)
    grad_a[:,-1]=np.dot(grad_h[:,-1],np.eye(m)*(1-np.tanh(A[:,-1])**2))
    for t in reversed(range(seq_lenght-1)):
        grad_h[:,t]=np.dot(G[t,:],rnn.V)+np.dot(grad_a[:,t+1],rnn.W)
        grad_a[:,t]=np.dot(grad_h[:,t],np.eye(m)*(1-np.tanh(A[:,t])**2))

    H[:,1:]=H[:,:-1]
    H[:,0]=h_prev[:,0]

    grad_b=np.sum(grad_a,axis=1).reshape(-1,1)
    grad_W=np.dot(grad_a,H.T)
    grad_U=np.dot(grad_a,X.T)
    return [grad_V, grad_W, grad_U, grad_c, grad_b], compute_loss(X,rnn,Y,h_prev), next_h

def clip_gradients(grad_list):
    for g in range(len(grad_list)):
        grad_list[g]=np.maximum(np.minimum(grad_list[g],5),-5)
    return grad_list

def train(rnn,book_data, seq_lenght, chars_with_indices, alph_size, hidden_size,n_epochs,lr):

    h_prev = np.zeros((hidden_size, 1))
    e=np.random.randint(0,len(book_data)-seq_lenght)
    X_chars = book_data[e:e + seq_lenght]
    Y_chars = book_data[e + 1:e + seq_lenght + 1]
    X = seq_to_ohm(alph_size, X_chars, chars_with_indices)
    Y = seq_to_ohm(alph_size, Y_chars, chars_with_indices)
    smooth_loss =compute_loss(X,rnn,Y,h_prev)

    smooth_loss_plot=[]
    iterations=0

    weights = rnn.rnn_to_list()
    momentums=rnn.initial_momentum()

    for epoch in range(n_epochs):
        print(epoch)
        h_prev = np.zeros((hidden_size, 1))
        e=0 # chars read so far
        while e< len(book_data)-seq_lenght:

            X_chars = book_data[e:e+seq_lenght]
            Y_chars = book_data[e+1:e+seq_lenght + 1]
            X = seq_to_ohm(alph_size, X_chars, chars_with_indices)
            Y = seq_to_ohm(alph_size, Y_chars, chars_with_indices)

            gradients,loss,h_prev=compute_gradients(X,Y,rnn,h_prev)
            gradients=clip_gradients(gradients)
            for i in range(len(gradients)):
                momentums[i]=momentums[i]+(gradients[i])**2
                weights[i]=weights[i]-((lr)/(np.sqrt(momentums[i]+1e-8))*gradients[i])

            rnn.update(weights)
            smooth_loss=0.999*smooth_loss+0.001*loss

            if iterations%100==0:
                smooth_loss_plot.append(smooth_loss)
            if iterations%1000==0:
                print(iterations)
                print(smooth_loss)
            if iterations%10000==0:
                my_string = generate_chars(chars_with_indices, alph_size, X_chars[0], rnn, 200, h_prev)
                print(my_string)
            e = e + seq_lenght
            iterations += 1

    plt.plot(smooth_loss_plot)
    plt.show()


h=1e-5# for numerical gradients

m=100 #dimensionality of the hidden state
seq_length=25 #length of the input sequence
chars_with_indices,book_data=read_data(INPUT_FILE)
K=np.size(chars_with_indices[:,0]) #alphabet size
lr=0.001


rnn=RNN(m,K)

#my_string=generate_chars(chars_with_indices,K,m,'a',rnn,n)

X_chars=book_data[0:seq_length]
Y_chars=book_data[1:seq_length+1]

X=seq_to_ohm(K,X_chars,chars_with_indices)
Y=seq_to_ohm(K,Y_chars,chars_with_indices)

"""h_0=np.random.normal(0,0.5,(m,1))
gradients_num=compute_grad_num_slow(X,Y,rnn,h,h_0)
gradients=compute_gradients(X,Y,rnn,h_0)
gradients_good=[]
for g in range(len(gradients)):
    gradients_good.append(check_grad(gradients[g], gradients_num[g],1e-6))"""
train(rnn,book_data,seq_length,chars_with_indices,K,m,7,lr)
a=0