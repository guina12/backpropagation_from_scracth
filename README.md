## 1- Backpropagation from scratch

  Currently, there are many deep learning frameworks that are used to deal with the complexity of gradient calculations
  We know that before it was very difficult to train neural networks, but today we have many techniques that we can use to have a more efficient training
  Even though we have several techniques and several frameworks available to train neural networks, it is always important to have a sense of what happens behind the frameworks, this 
 ensures that we have full control of what we are actually doing .

 1.1  - The main idea of this project is to show how we can correctly perform backpropagation without any framework, which is really complicated and difficult. In the real world, this is 
      not typically done, but when we learn and understand how things work under the hood, everything becomes much clearer.

 1.2 - Forward Pass

   ``` Python

    # forward pass , "chunked" into smaller steps that are  possible to  backward one at time
    emb=C[Xb] #embed the characters into vectors
    embcat=emb.view(emb.shape[0],-1) # concatenate the vectores
    # Layer 1
    hprebn=embcat @ W1+ b1
    # Batchnorm layer
    bnmeani = 1/n*hprebn.sum(axis=0,keepdim=True)
    bndiff = hprebn-bnmeani
    bndiff2 = bndiff**2
    bnvar = 1/(n-1)*(bndiff2).sum(axis=0,keepdim=True)
    bnvar_inv = (bnvar + 1e-5)**-0.5
    bnraw = bndiff * bnvar_inv
    hpreact = bngain*bnraw + bnbias
    # Non - linearity
    h = torch.tanh(hpreact) # hidden layer
    # Linear layer 2
    logits = h @ W2+b2 # output layer
    # cross_entropy loss (same as F.cross_entropy(logits,Yb))
    logits_maxes = logits.max(1,keepdim=True).values
    norm_logits  = logits - logits_maxes # subtract max for numerical
    counts  = norm_logits.exp()
    counts_sum = counts.sum(1,keepdims=True)
    counts_sum_inv = counts_sum**-1
    probs=counts * counts_sum_inv
    logprobs=probs.log()
    loss=-logprobs[range(n),Yb].mean()
    
    # Pytorch backward pass
    
    for p in parameters:
      p.grad = None
    for t in [logprobs,probs,counts,counts_sum,counts_sum_inv,
              norm_logits,logits_maxes,logits,h,hpreact,bnraw,
              bnvar,bnvar_inv, bndiff2,bndiff,hprebn,bnmeani,
              embcat,emb]:
              t.retain_grad()
    loss.backward()
```
1.2 - Backward pass

   ``` Python

    dlogprobs = torch.zeros_like(logprobs)
    dlogprobs[range(n),Yb] = -1.0/n
    dprobs = (1.0 / probs) * dlogprobs
    dcounts_sum_inv = (counts * dprobs).sum(1,keepdim=True)
    dcounts = counts_sum_inv * dprobs
    dcounts_sum  = (-counts_sum**-2) * dcounts_sum_inv
    dcounts += torch.ones_like(counts) * dcounts_sum
    dnorms_logits = counts * dcounts
    dlogits=dnorms_logits.clone()
    dlogits_maxes=(-dnorms_logits).sum(1,keepdims=True)
    dlogits += F.one_hot(logits.max(1).indices,num_classes = logits.shape[1] ) * dlogits_maxes
    dh = dlogits @ W2.T
    dw2 = h.T @ dlogits
    db2 = dlogits.sum(axis=0)
    dhpreact = (1.0 -h**2 ) * dh
    dbgain = (bnraw * dhpreact).sum(0,keepdims=True)
    dbnraw = bngain * dhpreact
    dbnbias = dhpreact.sum(0,keepdims=True)
    dbndiff = bnvar_inv * dbnraw
    dbnvar_inv = (bndiff * dbnraw).sum(0,keepdim=True)
    dbnvar = (-0.5*(bnvar + 1e-5)**-1.5) * dbnvar_inv
    dbndiff2 = (1.0 / (n-1)) * torch.ones_like(bndiff2) * dbnvar
    dbndiff += (2 * bndiff) * dbndiff2
    dhprebn = dbndiff.clone()
    dbmeani = (-dbndiff).sum(0)
    dhprebn += 1.0/n * (torch.ones_like(hprebn) * dbmeani)
    dembcat = dhprebn @ W1.T
    dw1 = embcat.T @ dhprebn
    db1 = dhprebn.sum(axis=0)
    demb = dembcat.view(emb.shape)
    dC = torch.zeros_like(C)
    for k in range(Xb.shape[0]):
      for j in range(Xb.shape[1]):
        ix = Xb[k,j]
        dC[ix] +=demb[k,j]
      

 

