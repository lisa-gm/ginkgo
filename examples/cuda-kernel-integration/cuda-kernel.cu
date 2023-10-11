constexpr unsigned int default_block_size = 512;


// Parallel Selected Inverse using Fixed Point Iter
// We try in-place updates instead of Jacobi-style updates
__global__ void parsinv_kernel( 
    int n, // matrix size
    int Lnnz, // number of nonzeros in LT stored in CSR, upper triangular  (equivalent to L in CSC)
    int *Lrowptr, // row pointer L
    int *Lcolidx, //col index L 
    double *Lval, // val array L
    int Snnz, // number of nonzeros in S (stored in CSR, full sparse)
    int *Srowptr, // row pointer S
    int *Srowidx, // row index S 
    int *Scolidx, //col index S 
    double *Sval // val array S 
    ){
    
    int threadidx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (threadidx < Snnz) {
        int i, j, il, is, jl, js;
        double Lii, s, sp;

        // handle element S(threadidx) = S(i,j)
        i = Srowidx[ threadidx ];
        j = Scolidx[ threadidx ];

        if( i>j ){
            // swap indices - there might be a more efficient way, though
            int t = i;
            i = j;
            j = t;
        }

        // retrieve L(i,i), easy as these are the first element in each row
        Lii = Lval[ Lrowptr[ i ] ];


        // we are working on a symmetric matrix S
        // if we notice j>i, we compute S(j,i) instead of S(i,j)
        // maybe later



        // compute L(i,:).* S(j,:)
        // il and is are iterating over the nonzero entries in the respective rows
        il = Lrowptr[ i ];
        is = Srowptr[ j ];
        while( il < Lrowptr[i+1] && is < Srowptr[ j+1 ] ){
            sp = 0.0;
            // jl and js are the col-indices of the respective nonzero entries
            jl = Lcolidx[ il ];
            js = Scolidx[ is ];
            sp = (jl == js) ? Lval[ il ] * Sval[ is ] : sp;
            s = (jl == js) ? s-sp : s;
            il = (jl <= js) ? il+1 : il;
            is = (jl >= js) ? is+1 : is;
        }
        s += sp;  // undo the last operation (it must be the last)    
        s = 1. / Lii * s; // scaling
        
        if (i == j) // diagonal element
            Sval[ threadidx ] = 1. / ( Lii * Lii) - s;
        else  
            Sval[ threadidx ] = - s;



    }
}



void parsinv(
    int n, // matrix size
    int Lnnz, // number of nonzeros in LT stored in CSR, upper triangular  (equivalent to L in CSC)
    int *Lrowptr, // row pointer L
    int *Lcolidx, //col index L
    double *Lval, // val array L
    int Snnz, // number of nonzeros in S (stored in CSR, full sparse)
    int *Srowptr, // row pointer S
    int *Srowidx, // row index S
    int *Scolidx, //col index S
    double *Sval // val array S
    ){

    unsigned int grid_dim = (Snnz + default_block_size - 1) / default_block_size;
    for( int i=0; i<10; i++){
    parsinv_kernel<<<dim3(grid_dim), dim3(default_block_size)>>>(n, Lnnz, Lrowptr, Lcolidx, Lval, Snnz, Srowptr, Srowidx, Scolidx, Sval);
    }
}
