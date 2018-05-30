import json
import numpy as np
import matplotlib.pyplot as plt

class GRU():
    """
    RNN class
    """
    def __init__(self,m,K):
        """
        Initialize the rnn with weights
        :param m: dimensionality of hidden state
        :param K: aplhabet size
        """
        self.V = np.random.normal(0, 0.1, (K, m))
        self.c = np.zeros((K, 1))  # bias vector, dimension (K,1), K is alphabet size

        self.W_h = np.random.normal(0, 0.1, (m, K))
        self.U_h = np.random.normal(0, 0.1, (m, m))
        self.b_h = np.zeros((m, 1))

        self.W_z = np.random.normal(0, 0.1, (m, K))
        self.U_z = np.random.normal(0, 0.1, (m, m))
        self.b_z = np.zeros((m, 1))

        self.W_r = np.random.normal(0, 0.1, (m, K))
        self.U_r = np.random.normal(0, 0.1, (m, m))
        self.b_r=np.zeros((m,1)) # bias vector, dimension (m,1), m is hidden state dimensionality

        #random initialization of weight matrices

    def update(self,weights):
        self.V=weights[0]
        self.c=weights[1]
        self.W_h=weights[2]
        self.U_h=weights[3]
        self.b_h=weights[4]
        self.W_z=weights[5]
        self.U_z=weights[6]
        self.b_z=weights[7]
        self.W_r=weights[8]
        self.U_r=weights[9]
        self.b_r=weights[10]

    def gru_to_list(self):
        """
        :return: a list with the RNN weights
        """
        weights = [self.V, self.c,self.W_h,self.U_h,self.b_h,self.W_z,self.U_z,self.b_z ,self.W_r , self.U_r,self.b_r]
        return weights

    def initial_momentum(self):
        """
        set the initial momentum to zero, and return the momentum for all the weights in one list
        """
        m_V=np.zeros_like((self.V))
        m_c = np.zeros_like((self.c))
        m_W_h = np.zeros_like((self.W_h))
        m_U_h = np.zeros_like((self.U_h))
        m_b_h = np.zeros_like((self.b_h))
        m_W_z = np.zeros_like((self.W_z))
        m_U_z = np.zeros_like((self.U_z))
        m_b_z = np.zeros_like((self.b_z))
        m_W_r = np.zeros_like((self.W_r))
        m_U_r = np.zeros_like((self.U_r))
        m_b_r = np.zeros_like((self.b_r))

        return [m_V,m_c,m_W_h,m_U_h,m_b_h,m_W_z,m_U_z,m_b_z,m_W_r,m_U_r,m_b_r]

def read_data(file_path):
    """

    :param file_path: path of the input file
    :return: an array of characters and associated indices, and all the characters in the book
    """

    with open(file_path,"r") as input:
        data=input.read()

    chars=list(set(data))
    chars.sort()
    indices=np.arange(len(chars))
    char_and_ind=np.asarray(list(zip(chars,indices)))
    return char_and_ind,data

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
    """converts a sequence of chars in a onehot matrix"""
    matrix=np.zeros((alph_size,len(sequence)),dtype=int)
    for c,char in enumerate(sequence):
        matrix[index_from_char(chars_with_indices, char ),c]+=1
    return matrix

def sigmoid(x):
    return np.exp(x)/(np.exp(x)+1)

def softmax(S):
    return np.exp(S)/np.sum(np.exp(S),axis=0)


def forward(X,gru,h_prev):
    """forwars pass"""
    h_0=h_prev
    a= np.zeros((gru.b_r.shape[0], np.shape(X)[1]))
    b=np.zeros_like(a)
    c=np.zeros_like(a)
    r=np.zeros_like(a)
    z=np.zeros_like(b)
    s=np.zeros_like(c)
    h=np.zeros_like(s)
    probabilities=np.zeros((X.shape))
    for t,x in enumerate(X.T):
        if t==0:
            a[:,t] = (gru.W_r @ x.reshape(-1,1) + gru.U_r @ h_0 + gru.b_r)[:,0]
            b[:, t] = (gru.W_z @ x.reshape(-1, 1) + gru.U_z @ h_0 + gru.b_z)[:, 0]
            r[:,t]=sigmoid(a[:,t])
            z[:, t] = sigmoid(b[:, t])
            c[:,t]=(gru.W_h @ x.reshape(-1,1) + gru.U_h @ (h_0*r[:,t].reshape(-1,1)) + gru.b_h)[:,0]
            s[:, t] = np.tanh(c[:, t])
            h[:, t] = ((1 - z[:, t].reshape(-1,1)) * h_0 + z[:,t].reshape(-1,1) * s[:,t].reshape(-1,1))[:,0]
        else:
            a[:, t] = (gru.W_r @ x.reshape(-1, 1) + gru.U_r @ h[:,t-1].reshape(-1,1) + gru.b_r)[:, 0]
            b[:, t] = (gru.W_z @ x.reshape(-1, 1) + gru.U_z @ h[:,t-1].reshape(-1,1) + gru.b_z)[:, 0]
            r[:, t] = sigmoid(a[:, t])
            z[:, t] = sigmoid(b[:, t])
            c[:, t] = (gru.W_h @ x.reshape(-1, 1) + gru.U_h @ (h[:,t-1].reshape(-1,1)* r[:, t].reshape(-1, 1)) + gru.b_h)[:, 0]
            s[:, t] = np.tanh(c[:, t])
            h[:, t] = ((1 - z[:, t].reshape(-1,1)) * h[:,t-1].reshape(-1,1) + z[:, t].reshape(-1,1) * s[:, t].reshape(-1,1))[:,0]


        o_t = gru.V @ h[:,t].reshape(-1,1) + gru.c
        p_t = softmax(o_t)
        probabilities[:,t]=p_t[:,0]

    return probabilities,{'a':a,'b':b,'c':c,'r':r,'z':z,'s':s,'h':h}


def generate_chars(chars_with_indices,alph_size,m,first_char,gru,n):
    """
    Generate synthetic text
    :param chars_with_indices: array of chars coupled with an index
    :param alph_size: alhabet size
    :param first_char: char to begin with
    :param rnn: the RNN object
    :param n: Length of text to generate
    :param h_0: initial hidden state
    :return: the generated chars
    """
    labels = [first_char]
    h = np.zeros((m, 1))
    for t in range(n-1):
        x = to_onehot(alph_size, chars_with_indices, labels[-1])
        a = gru.W_r @ x.reshape(-1, 1) + gru.U_r @ h + gru.b_r
        b = gru.W_z @ x.reshape(-1, 1) + gru.U_z @ h + gru.b_z
        r = sigmoid(a)
        z = sigmoid(b)
        c = gru.W_h @ x.reshape(-1, 1) + gru.U_h @ (h * r) + gru.b_h
        s = np.tanh(c)
        h = (1 - z) * h + z * s
        o_t = gru.V @ h + gru.c
        p_t = softmax(o_t)
        labels.append(char_from_index(chars_with_indices,np.where(np.random.multinomial(1, p_t[:, 0]) == 1)[0][0]))

    return ''.join(labels)

def compute_loss(X,gru,Y,h_prev):
    P, states=forward(X,gru,h_prev)
    loss=-np.sum(np.log(np.sum(Y * P, axis=0)))
    return loss, P, states

def check_grad(grad_a, grad_n, eps):
    '''function to compare the analitical (grad_a) and numerical (grad_n) gradients'''
    diff = np.abs(grad_a - grad_n) / max(eps, np.amax(np.abs(grad_a) + np.abs(grad_n)))
    if np.amax(diff) < 1e-6:
        return True
    else:
        return False


def compute_grad_num_slow(X, Y, gru,h,h_prev):
    '''centered difference gradient for W and Fs'''
    weights=gru.gru_to_list()
    gradients=[]
    for weight in weights:
        grad_w=np.zeros_like(weight)
        it = np.nditer(weight, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            iw = it.multi_index
            old_value = weight[iw]
            weight[iw] = old_value - h  # use original value
            c1,_,_ = compute_loss(X,gru,Y,h_prev)
            weight[iw] = old_value + h  # use original value
            c2,_,_ = compute_loss(X, gru, Y,h_prev)
            grad_w[iw] = (c2 - c1) / (2 * h)
            weight[iw] = old_value  # restore original value
            it.iternext()
        gradients.append(grad_w)

    return gradients

def clip_gradients(grad_list):
    for g in range(len(grad_list)):
        grad_list[g]=np.where(grad_list[g]>5,5,grad_list[g])
        grad_list[g] = np.where(grad_list[g] < -5, -5, grad_list[g])
        #grad_list[g]=np.maximum(np.minimum(grad_list[g],5),-5)
    return grad_list

def compute_gradients(X,Y,gru,h_prev):
    loss,P, states=compute_loss(X,gru,Y,h_prev)
    R=states['r']
    Z=states['z']
    S=states['s']
    H=states['h']
    m=np.shape(H)[0] #100
    seq_length=np.shape(H)[1] #25
    next_h = H[:, -1].reshape(-1, 1).copy()
    grad_a = np.zeros((m, seq_length))
    grad_h = np.zeros((m, seq_length))
    grad_s = np.zeros_like(grad_a)
    grad_z = np.zeros_like(grad_a)
    grad_r = np.zeros_like(grad_a)
    grad_c = np.zeros_like(grad_a)
    grad_b = np.zeros_like(grad_a)

    G=-(Y-P).T
    grad_V=G.T @ H.T
    grad_cc = np.sum(G, axis=0).reshape(-1, 1)
    H[:, 1:] = H[:, :-1]
    H[:, 0] = h_prev[:, 0]

    grad_h[:,-1]=G[-1,:] @ gru.V
    grad_s[:,-1]=grad_h[:,-1] * Z[:,-1]
    grad_c[:,-1]=grad_s[:,-1] * (1-S[:,-1]**2)
    grad_z[:,-1]=grad_h[:,-1] * (-H[:,-1]+S[:,-1])
    grad_b[:,-1]=grad_z[:,-1] * (Z[:,-1]*(1-Z[:,-1]))
    grad_r[:,-1]=grad_c[:,-1] @ (gru.U_h*H[:,-1])
    grad_a[:,-1]= grad_r[:,-1] * (R[:,-1]*(1-R[:,-1]))
    for t in reversed(range(seq_length-1)):
        grad_h[:, t] = G[t, :] @ gru.V + grad_h[:, t + 1] * (1 - Z[:, t + 1]) + \
                       grad_c[:, t + 1] @ (gru.U_h*R[:,t + 1]) + \
                       grad_b[:, t + 1] @ gru.U_z + grad_a[:, t + 1] @ gru.U_r
        grad_s[:, t] = grad_h[:, t] *(Z[:, t])
        grad_c[:, t] = grad_s[:, t] *(1 - S[:,t] ** 2)
        grad_z[:, t] = grad_h[:, t] *(-H[:,t] + S[:, t])
        grad_b[:, t] = grad_z[:, t] *(Z[:,t] * (1 - Z[:,t]))
        grad_r[:, t] = grad_c[:, t] @(gru.U_h*H[:,t])
        grad_a[:, t] = grad_r[:, t] *(R[:,t]* (1 - R[:,t]))


    grad_bh=np.sum(grad_c,axis=1).reshape(-1,1)
    grad_Wh=grad_c @ X.T
    grad_Uh=grad_c @ (R*H).T
    grad_bz=np.sum(grad_b,axis=1).reshape(-1,1)
    grad_Wz=grad_b @ X.T
    grad_Uz=grad_b @ H.T
    grad_br = np.sum(grad_a, axis=1).reshape(-1, 1)
    grad_Wr = grad_a @ X.T
    grad_Ur = grad_a @ H.T
    return [grad_V, grad_cc,grad_Wh, grad_Uh , grad_bh, grad_Wz, grad_Uz, grad_bz, grad_Wr, grad_Ur ,grad_br], loss, next_h

def train(gru,oh_tweets, seq_length, chars_with_indices, alph_size, hidden_size,n_epochs,lr):

    book_length = np.shape(oh_book)[1]
    h_prev = np.zeros((hidden_size, 1))
    e = np.random.randint(0, len(book_data) - seq_length)
    X = oh_book[:, e:e + seq_length]
    Y = oh_book[:, e + 1:e + seq_length + 1]
    smooth_loss = compute_loss(X, gru, Y, h_prev)[0]

    smooth_loss_plot = []
    iterations = 0

    weights = gru.gru_to_list()
    momentums=gru.initial_momentum()

    for epoch in range(n_epochs):
        print(epoch)
        h_prev = np.zeros_like(h_prev)
        e=0 # chars read so far
        while e< book_length-seq_length:

            X = oh_book[:, e:e + seq_length]
            Y = oh_book[:, e + 1:e + seq_length + 1]

            gradients,loss,h_prev=compute_gradients(X,Y,gru,h_prev)
            gradients=clip_gradients(gradients)
            for i in range(len(gradients)):
                momentums[i]=momentums[i]+(gradients[i])**2
                weights[i]=weights[i]-((lr)/(np.sqrt(momentums[i]+1e-6))*gradients[i])

            gru.update(weights)
            smooth_loss=0.999*smooth_loss+0.001*loss

            if iterations%100==0:
                smooth_loss_plot.append(smooth_loss)
            if iterations%1000==0:
                print(iterations)
                print(smooth_loss)
            if iterations%10000==0:
                my_string = generate_chars(chars_with_indices, alph_size,hidden_size, char_from_index(chars_with_indices,np.argwhere(X[:,-1]==1)[0][0]), gru, 200)
                print(my_string)
            e = e + seq_length
            iterations += 1

    plt.plot(smooth_loss_plot)

    my_string = generate_chars(chars_with_indices, alph_size,hidden_size, 'H', gru, 1000)
    print(my_string)
    plt.show()



INPUT_FILE='goblet_book.txt'


h=1e-4# for numerical gradients

m=100 #dimensionality of the hidden state
seq_length=25 #length of the input sequence
chars_with_indices,book_data=read_data(INPUT_FILE)
K=np.size(chars_with_indices[:,0]) #alphabet size
lr=0.1

oh_book=seq_to_ohm(K,book_data,chars_with_indices)


np.random.seed(100)
gru=GRU(m,K)
'''gradients,loss,h_prev=compute_gradients(X,Y,gru,h_0)
gradients_num=compute_grad_num_slow(X,Y,gru,h,h_0)
gradients_good=[]
for g in range(len(gradients)):
    gradients_good.append(check_grad(gradients[g], gradients_num[g],1e-6))
'''
train(gru,oh_book,seq_length,chars_with_indices,K,m,20,lr)

