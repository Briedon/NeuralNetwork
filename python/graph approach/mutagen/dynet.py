import dynet as dy

## ==== Create a new computation graph
# (it is a singleton, we have one at each stage.
# dy.renew_cg() clears the current one and starts anew)
print(dy.DeviceInfo)

## ==== Creating Expressions from user input / constants.
x = dy.scalarInput(value)

v = dy.vecInput(dimension)
v.set([1,2,3])

z = dy.matInput(dim1, dim2)

# for example:
z1 = dy.matInput(2, 2)
z1.set([1,2,3,4]) # Column major

# Or directly from a numpy array
z1 = dy.inputTensor([[1,2],[3,4]]) # Row major

## ==== We can take the value of an expression.
# For complex expressions, this will run forward propagation.
print(z.value())


## ==== Parameters
# Parameters are things we tune during training.
# Usually a matrix or a vector.

# First we create a parameter collection and add the parameters to it.
m = ParameterCollection()
pW = m.add_parameters((8,8)) # an 8x8 matrix
pb = m.add_parameters(8)

# then we create an Expression out of the parameter collection's parameters
W = dy.parameter(pW)
b = dy.parameter(pb)

## ===== Lookup parameters
# Similar to parameters, but are representing a "lookup table"
# that maps numbers to vectors.
# These are used for embedding matrices.
# for example, this will have VOCAB_SIZE rows, each of DIM dimensions.
lp  = m.add_lookup_parameters((VOCAB_SIZE, DIM))

# lookup parameters can be initialized from an existing array, i.e:
# m["lookup"].init_from_array(wv)

e5  = dy.lookup(lp, 5)   # create an Expression from row 5.
e5  = lp[5]           # same
e5c = dy.lookup(lp, 5, update=False)  # as before, but don't update when optimizing.

e5  = dy.lookup_batch(lp, [4, 5])   # create a batched Expression from rows 4 and 5.
e5  = lp.batch([4, 5])           # same

e5.set(9)   # now the e5 expression contains row 9
e5c.set(9)  # ditto


## ===== Combine expression into complex expressions.

# Math
e = e1 + e2
e = e1 * e2   # for vectors/matrices: matrix multiplication (like e1.dot(e2) in numpy)
e = e1 - e2
e = -e1

e = dy.dot_product(e1, e2)
e = dy.cmult(e1, e2)           # component-wise multiply  (like e1*e2 in numpy)
e = dy.cdiv(e1, e2)            # component-wise divide
e = dy.colwise_add(e1, e2)     # column-wise addition

# Matrix Shapes
e = dy.reshape(e1, new_dimension)
e = dy.transpose(e1)

# Per-element unary functions.
e = dy.tanh(e1)
e = dy.exp(e1)
e = dy.log(e1)
e = dy.logistic(e1)   # Sigmoid(x)
e = dy.rectify(e1)    # Relu (= max(x,0))
e = dy.softsign(e1)    # x/(1+|x|)

# softmaxes
e = dy.softmax(e1)
e = dy.log_softmax(e1, restrict=[]) # restrict is a set of indices.
                                 # if not empty, only entries in restrict are part
                                 # of softmax computation, others get 0.


e = dy.sum_cols(e1)


# Picking values from vector expressions
e = dy.pick(e1, k)              # k is unsigned integer, e1 is vector. return e1[k]
e = e1[k]                    # same

e = dy.pickrange(e1, k, v)      # like python's e1[k:v] for lists. e1 is an Expression, k,v integers.
e = e1[k:v]                  # same

e = dy.pickneglogsoftmax(e1, k) # k is unsigned integer. equiv to: (pick(-log(dy.softmax(e1)), k))


# Neural net stuff
dy.noise(e1, stddev) # add a noise to each element from a gausian with standard-dev = stddev
dy.dropout(e1, p)    # apply dropout with probability p

# functions over lists of expressions
e = dy.esum([e1, e2, ...])            # sum
e = dy.average([e1, e2, ...])         # average
e = dy.concatenate_cols([e1, e2, ...])  # e1, e2,.. are column vectors. return a matrix. (sim to np.hstack([e1,e2,...])
e = dy.concatenate([e1, e2, ...])     # concatenate

e = dy.affine_transform([e0,e1,e2, ...])  # e = e0 + ((e1*e2) + (e3*e4) ...)

## Loss functions
e = dy.squared_distance(e1, e2)
e = dy.l1_distance(e1, e2)
e = dy.huber_distance(e1, e2, c=1.345)

# e1 must be a scalar that is a value between 0 and 1
# e2 (ty) must be a scalar that is a value between 0 and 1
# e = ty * log(e1) + (1 - ty) * log(1 - e1)
e = dy.binary_log_loss(e1, e2)

# e1 is row vector or scalar
# e2 is row vector or scalar
# m is number
# e = max(0, m - (e1 - e2))
e = dy.pairwise_rank_loss(e1, e2, m=1.0)

# Convolutions
# e1 \in R^{d x s} (input)
# e2 \in R^{d x m} (filter)
e = dy.conv1d_narrow(e1, e2) # e = e1 *conv e2
e = dy.conv1d_wide(e1, e2)   # e = e1 *conv e2
e = dy.filter1d_narrow(e1, e2) # e = e1 *filter e2

e = dy.kmax_pooling(e1, k) #  kmax-pooling operation (Kalchbrenner et al 2014)
e = dy.kmh_ngram(e1, k) #
e = dy.fold_rows(e1, nrows=2) #
