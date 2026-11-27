def embedding_lookup(ids, table):
    return table[ids]                                 # gather instead of a one-hot matmul
