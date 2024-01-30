/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include <cstdlib>
#include <fstream>
#include <ginkgo/ginkgo.hpp>
#include <iostream>
#include <chrono>
#include <stdlib.h>
#include <stdio.h>

using value_type = double;
using index_type = gko::int32;
using Csr = gko::matrix::Csr<value_type, index_type>;
using Coo = gko::matrix::Coo<value_type, index_type>;
using Dense = gko::matrix::Dense<value_type>;

void parsinv(
    int n, // matrix size
    int Lnnz, // number of nonzeros in LT stored in CSR, upper triangular  (equivalent to L in CSC)
    const int *Lrowptr, // row pointer L
    const int *Lcolidx, //col index L
    const double *Lval, // val array L
    int Snnz, // number of nonzeros in S (stored in CSR, full sparse)
    const int *Srowptr, // row pointer S
    const int *Srowidx, // row index S
    const int *Scolidx, //col index S
    double *Sval, // val array S
    const int* sym_map
    );

void parsinv_residual(
    int n, // matrix size
    int Annz, // number of nonzeros in A
    int *Arowptr, // row pointer A
    int *Arowidx, //row index A
    int *Acolidx, //col index A
    double *Aval, // val array A
    int *Srowptr, // row pointer S
    int *Scolidx, //col index S
    double *Sval, // val array S
    double *tval
    );

void symmapping(int n,     // matrix size
             int Snnz,  // number of nonzeros in S (stored in CSR, full sparse)
             const int* Srowptr,  // row pointer S
             const int* Srowidx,  // row index S
             const int* Scolidx,  // col index S
             int* sym_map    // symmetric entry mapping
                                //contains in each location k corresponding
                                // to location(i,j) the entry t corresponding to (j,i)
	);

void ASpOnesB(
    int n, // matrix size
    int *Arowptr, // row pointer A
    int *Acolidx, //col index A
    double *Aval, // val array A
    const int *Browptr, // row pointer B
    const int *Bcolidx, //col index B
    const double *Bval // val array B
    );



inline void read_load_input(std::string inputName, std::shared_ptr<gko::CudaExecutor> gpu, std::unique_ptr<Csr> Matrix_csr, int flag){

    std::ifstream Matrix_csr_file(inputName);

    if(inputName.compare("trueInverse") == 0      || inputName.compare("TrueInverse") == 0){
        flag = 1;
        std::cout << " Computing error to reference solution in iterations "  << std::endl;
        Matrix_csr = gko::read<Csr>(Matrix_csr_file, gpu);

    } else if(inputName.compare("choleskyF") == 0 || inputName.compare("CholeskyF") == 0){
        flag = 1;
        std::cout << " Using externally provided Cholesky factor. Assuming A = L*t(L). "  << std::endl;
        Matrix_csr = gko::read<Csr>(Matrix_csr_file, gpu);

    } else if(inputName.compare("initialGuess") == 0 || inputName.compare("InitialGuess") == 0){
        flag = 1;
        std::cout << " Using externally provided initial guess for S. "  << std::endl;
        Matrix_csr = gko::read<Csr>(Matrix_csr_file, gpu);            
    }

}

int main(int argc, char* argv[])
{

    // Instantiate a CUDA executor
    std::shared_ptr<gko::CudaExecutor> gpu = gko::CudaExecutor::create(0, gko::OmpExecutor::create());
    
    // timers
    double factorization_time;
    std::chrono::time_point<std::chrono::steady_clock> start;
    std::chrono::time_point<std::chrono::steady_clock> end;

    // debug = 0 : only compute selected inverse based on matrix
    // debug = 1 : feed in reference solution and compare against solution
    int debug = 0;

    int extSinitial = 0;
    int extL = 0;

    std::ifstream A_file(argv[1]);
    auto A_csr = gko::share(gko::read<Csr>(A_file, gpu));

    // TODO: wrong type ... doesn't compile ...
    //std::unique_ptr<Csr> I;
    //std::unique_ptr<Csr> L;
    //std::unique_ptr<Csr> S_csr;

    std::unique_ptr<gko::matrix::Csr<value_type, index_type>> S_csr;
    std::unique_ptr<gko::matrix::Csr<value_type, index_type>> L;
    std::unique_ptr<gko::matrix::Csr<value_type, index_type>> I;

    // options are 2, 4, 6, 8 
    // ./cuda-kernel A choleskyF L trueInverse invA initialGuess S (order arbitrary)
    if (argc == (1 + 1) ) {
    }
    else if ( argc == (1 + 3) || argc == (1 + 5) || argc == (1 + 7)){

        std::string inputName2 = argv[2];
        if(inputName2.compare("trueInverse") == 0        || inputName2.compare("TrueInverse") == 0){
            read_load_input(inputName2, gpu, I, debug);
        } else if(inputName2.compare("choleskyF") == 0   || inputName2.compare("CholeskyF") == 0){
            read_load_input(inputName2, gpu, L, extL);
        } else if(inputName2.compare("initialGuess") == 0 || inputName2.compare("InitialGuess") == 0){
            read_load_input(inputName2, gpu, S_csr, extSinitial);
        } else {
            printf("invalid input name. Options are: trueInverse choleskyF initialGuess");
        }

        if(argc == (1 + 5) || argc == (1 + 7)){
            std::string inputName4 = argv[4];

            if(inputName4.compare("trueInverse") == 0        || inputName4.compare("TrueInverse") == 0){
                read_load_input(inputName4, gpu, I, debug);
            } else if(inputName4.compare("choleskyF") == 0   || inputName4.compare("CholeskyF") == 0){
                read_load_input(inputName4, gpu, L, extL);
            } else if(inputName4.compare("initialGuess") == 0 || inputName4.compare("InitialGuess") == 0){
                read_load_input(inputName4, gpu, S_csr, extSinitial);
            } else {
                printf("invalid input name. Options are: trueInverse choleskyF initialGuess");
            }

            if(argc == (1 + 7)){
                std::string inputName6 = argv[6];

                if(inputName6.compare("trueInverse") == 0        || inputName6.compare("TrueInverse") == 0){
                    read_load_input(inputName6, gpu, I, debug);
                } else if(inputName6.compare("choleskyF") == 0   || inputName6.compare("CholeskyF") == 0){
                    read_load_input(inputName6, gpu, L, extL);
                } else if(inputName6.compare("initialGuess") == 0 || inputName6.compare("InitialGuess") == 0){
                    read_load_input(inputName6, gpu, S_csr, extSinitial);
                } else {
                    printf("invalid input name. Options are: trueInverse choleskyF initialGuess");
                }

            }

        }

    } else {
        std::cout << "Please execute with the following parameters:\n"
                  << argv[0]
                  << "<A matrix path> [optional: <I matrix path> ]\n";
        std::exit(1);
    }

    exit(1);

    // if extL == 0 && extSinitial == 0 -> compute Cholesky factor, store L and store L + L^T + diag(L) in S_csr
    // if extL == 0 && extSinitial == 1 -> compute Cholesky factor, store L => maybe perform check that sparsity pattern matches on lower part?
    // if extL == 1 && extSinitial == 0 -> store L + L^T + diag(L) in S_csr
    // if extL == 1 && extSinitial == 1 -> just perform check that sparsity patterns match
   
   // compute cholesky factor 
   // TODO: add reordering ... -> don't allow reordering if initial guess S is provided ... sparsity pattern depends on L ...
   // or figure out some other way ...
    /* auto P = gko::experimental::reorder::NestedDissection<value_type, index_type>::build().on(gpu)->generate(A_csr);
    A_csr = A_csr->permute(P);

    if( debug > 0 ){
        I = I->permute(P);
    }
    */

   if(extL == 0 && extSinitial == 0){
        start = std::chrono::steady_clock::now();
        // Cholesky factors stored as L + L^T - diag(L)
        S_csr = gko::experimental::factorization::Cholesky<value_type, index_type>::build().on(gpu)->generate(A_csr);
        auto LLU = S_csr->unpack();
        L = LLU->get_upper_factor();
        end = std::chrono::steady_clock::now();
        factorization_time = std::chrono::duration<double>(end-start).count();

    } else {
        if(extSinitial == 0){
            // TODO: store L + L^T + diag(L) in S_csr using L
        }
    }

    if(extSinitial == 1){
        // TODO: check that sparsity pattern matches.
    }


    // TODO: not sure what this is doing? copy data to GPU?
    const auto num_row_ptrs = S_csr->get_size()[0] + 1;
    gko::array<index_type> row_ptrs_array(gpu, num_row_ptrs);
    gpu->copy_from(gpu, num_row_ptrs, S_csr->get_combined()->get_const_row_ptrs(),
                row_ptrs_array.get_data());

    auto S_coo = Coo::create(gpu);
    // write S_csr into S_coo -> contains L + L^T - diag(L)
    S_csr->get_combined()->convert_to(S_coo);

    //gko::write(std::ofstream{"cholesky.mtx"}, S_coo);

    auto S_row_ptrs = row_ptrs_array.get_const_data();
    auto S_row_idxs = S_coo->get_const_row_idxs();
    auto S_col_idxs = S_coo->get_const_col_idxs();
    auto S_values = S_coo->get_values();

    // create a vector for the symmetric mapping in S
    gko::array<int> sym_map(gpu, S_coo->get_num_stored_elements());
    symmapping( S_coo->get_size()[0],
                    S_coo->get_num_stored_elements(),
                    S_row_ptrs,
                    S_row_idxs,
                    S_col_idxs,
                    sym_map.get_data());


    // if extSinitial == 1
   
    // compute error to correct solution
    auto size = S_coo->get_num_stored_elements();
    auto neg_one = gko::initialize<Dense>({-gko::one<value_type>()}, gpu);
    std::unique_ptr<const Dense>A_vec;
    std::unique_ptr<const Dense>B_vec;
    if( debug > 0 ){
    // could also use sparsity pattern of A instead of AselInv
	auto sp_size = I->get_num_stored_elements();
	A_vec = Dense::create_const(
            gpu, gko::dim<2>{sp_size, 1},
            gko::array<value_type>::const_view(gpu, 
		    sp_size, I->get_const_values()), 1);
	auto ASelInv = I->clone();
    // 
	ASpOnesB( ASelInv->get_size()[0], 
			ASelInv->get_row_ptrs(),
			ASelInv->get_col_idxs(),
			ASelInv->get_values(),
			S_row_ptrs,
			S_col_idxs,
			S_values);
        // view selInv entries into vector to compute norm
        B_vec = Dense::create_const(
            gpu, gko::dim<2>{sp_size, 1},
            gko::array<value_type>::const_view(gpu, 
		    sp_size, ASelInv->get_const_values()), 1);
    
    	auto result =
           gko::matrix::Dense<value_type>::create(gpu, gko::dim<2>{1, 1});
    	A_vec->compute_norm2(result);
    	std::cout << "norm selected inverse : "
                  << gpu->copy_val_to_host(result->get_values()) << std::endl;
	
    	B_vec->compute_norm2(result);
    	std::cout << "norm initial guess : "
                 << gpu->copy_val_to_host(result->get_values()) << std::endl;

    	auto work_vec = A_vec->clone();
    	work_vec->add_scaled(neg_one, B_vec);
    	work_vec->compute_norm2(result);
    	printf("Frobenious norm iteration %2d: %.4e\n",
                  0, gpu->copy_val_to_host(result->get_values()));
    }
    // end error computation
    
    double inverse_time = 0.0;
    // Solve system
    for(int i=0; i<50; i++){
    	start = std::chrono::steady_clock::now();
	parsinv(L->get_size()[0], 
		    L->get_num_stored_elements(), 
		    L->get_const_row_ptrs(), 
		    L->get_const_col_idxs(),
		    L->get_const_values(),
            S_coo->get_num_stored_elements(),
		    S_row_ptrs,
		    S_row_idxs,
		    S_col_idxs,
		    S_values,
		    sym_map.get_const_data()
    	);
	gpu->synchronize();
	end = std::chrono::steady_clock::now();
        inverse_time += std::chrono::duration<double>(end-start).count();
	if( debug > 0 ){
		// compute after every iteration the error to correct solution
	        auto sp_size = I->get_num_stored_elements(); 
		A_vec = Dense::create_const(
        	    gpu, gko::dim<2>{sp_size, 1},
            	gko::array<value_type>::const_view(gpu,
                    sp_size, I->get_const_values()), 1);
        	auto ASelInv = I->clone();
        	ASpOnesB( ASelInv->get_size()[0],
                        ASelInv->get_row_ptrs(),
                        ASelInv->get_col_idxs(),
                        ASelInv->get_values(),
                        S_row_ptrs,
                        S_col_idxs,
                        S_values);
        	B_vec = Dense::create_const(
            		gpu, gko::dim<2>{sp_size, 1},
            	gko::array<value_type>::const_view(gpu,
                    sp_size, ASelInv->get_const_values()), 1);

    		auto result =
           		gko::matrix::Dense<value_type>::create(gpu, gko::dim<2>{1, 1});

    		auto work_vec = A_vec->clone();
    		work_vec->add_scaled(neg_one, B_vec);
    		work_vec->compute_norm2(result);
    		printf("Frobenious norm iteration %2d: %.4e\n",
                  i+1, gpu->copy_val_to_host(result->get_values()));
	}
    }

    printf("\n#####################################################################\n");
    printf("#\n# Factorization time: %.4e\n# Selected inverse time: %.4e\n#", factorization_time, inverse_time);
    printf("\n#####################################################################\n");

    // Write result
    //write(std::cout, I);
   //write(std::cout, S_coo);

   //std::ofstream sol_file = "selInv_mode.mtx";
   //write(sol_inv, S_coo->permute(P, gko::matrix::permute_mode::symm_inverse));
}
