/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

#include <algorithm>
#include <memory>
#include <numeric>
#include <random>
#include <vector>


#include <gtest/gtest.h>


#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/base/name_demangling.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/preconditioner/jacobi.hpp>
#include <ginkgo/core/solver/bicg.hpp>
#include <ginkgo/core/solver/bicgstab.hpp>
#include <ginkgo/core/solver/cb_gmres.hpp>
#include <ginkgo/core/solver/cg.hpp>
#include <ginkgo/core/solver/cgs.hpp>
#include <ginkgo/core/solver/fcg.hpp>
#include <ginkgo/core/solver/gmres.hpp>
#include <ginkgo/core/solver/idr.hpp>
#include <ginkgo/core/solver/ir.hpp>
#include <ginkgo/core/solver/lower_trs.hpp>
#include <ginkgo/core/solver/upper_trs.hpp>


#include "core/test/utils.hpp"
#include "test/utils/executor.hpp"


#if GINKGO_DPCPP_SINGLE_MODE
using solver_value_type = float;
#else
using solver_value_type = double;
#endif  // GINKGO_DPCPP_SINGLE_MODE


template <typename SolverType>
struct SimpleSolverTest {
    using solver_type = SolverType;
    using value_type = typename solver_type::value_type;
    using index_type = gko::int32;
    using matrix_type = gko::matrix::Dense<value_type>;

    static typename solver_type::parameters_type build(
        std::shared_ptr<const gko::Executor> exec,
        gko::size_type iteration_count)
    {
        return solver_type::build().with_criteria(
            gko::stop::Iteration::build()
                .with_max_iters(iteration_count)
                .on(exec));
    }

    static typename solver_type::parameters_type build_preconditioned(
        std::shared_ptr<const gko::Executor> exec,
        gko::size_type iteration_count)
    {
        return solver_type::build()
            .with_criteria(gko::stop::Iteration::build()
                               .with_max_iters(iteration_count)
                               .on(exec))
            .with_preconditioner(
                gko::preconditioner::Jacobi<value_type, index_type>::build()
                    .with_max_block_size(1u)
                    .on(exec));
    }

    static void assert_empty_state(const solver_type* mtx)
    {
        ASSERT_FALSE(mtx->get_size());
        ASSERT_EQ(mtx->get_system_matrix(), nullptr);
        ASSERT_EQ(mtx->get_preconditioner(), nullptr);
        ASSERT_EQ(mtx->get_stopping_criterion_factory(), nullptr);
    }
};


struct Cg : SimpleSolverTest<gko::solver::Cg<solver_value_type>> {};


struct Cgs : SimpleSolverTest<gko::solver::Cgs<solver_value_type>> {};


struct Fcg : SimpleSolverTest<gko::solver::Fcg<solver_value_type>> {};


struct Bicg : SimpleSolverTest<gko::solver::Bicg<solver_value_type>> {};


struct Bicgstab : SimpleSolverTest<gko::solver::Bicgstab<solver_value_type>> {};


struct Idr1 : SimpleSolverTest<gko::solver::Idr<solver_value_type>> {
    static typename solver_type::parameters_type build(
        std::shared_ptr<const gko::Executor> exec,
        gko::size_type iteration_count)
    {
        return solver_type::build()
            .with_criteria(gko::stop::Iteration::build()
                               .with_max_iters(iteration_count)
                               .on(exec))
            .with_subspace_dim(1u);
    }

    static typename solver_type::parameters_type build_preconditioned(
        std::shared_ptr<const gko::Executor> exec,
        gko::size_type iteration_count)
    {
        return solver_type::build()
            .with_criteria(gko::stop::Iteration::build()
                               .with_max_iters(iteration_count)
                               .on(exec))
            .with_preconditioner(
                gko::preconditioner::Jacobi<value_type, index_type>::build()
                    .with_max_block_size(1u)
                    .on(exec))
            .with_subspace_dim(1u);
    }
};


struct Idr4 : SimpleSolverTest<gko::solver::Idr<solver_value_type>> {
    static typename solver_type::parameters_type build(
        std::shared_ptr<const gko::Executor> exec,
        gko::size_type iteration_count)
    {
        return solver_type::build()
            .with_criteria(gko::stop::Iteration::build()
                               .with_max_iters(iteration_count)
                               .on(exec))
            .with_subspace_dim(4u);
    }

    static typename solver_type::parameters_type build_preconditioned(
        std::shared_ptr<const gko::Executor> exec,
        gko::size_type iteration_count)
    {
        return solver_type::build()
            .with_criteria(gko::stop::Iteration::build()
                               .with_max_iters(iteration_count)
                               .on(exec))
            .with_preconditioner(
                gko::preconditioner::Jacobi<value_type, index_type>::build()
                    .with_max_block_size(1u)
                    .on(exec))
            .with_subspace_dim(4u);
    }
};


struct Ir : SimpleSolverTest<gko::solver::Ir<solver_value_type>> {
    static typename solver_type::parameters_type build_preconditioned(
        std::shared_ptr<const gko::Executor> exec,
        gko::size_type iteration_count)
    {
        return solver_type::build()
            .with_criteria(gko::stop::Iteration::build()
                               .with_max_iters(iteration_count)
                               .on(exec))
            .with_solver(
                gko::preconditioner::Jacobi<value_type, index_type>::build()
                    .with_max_block_size(1u)
                    .on(exec));
    }
};


struct CbGmres2 : SimpleSolverTest<gko::solver::CbGmres<solver_value_type>> {
    static typename solver_type::parameters_type build(
        std::shared_ptr<const gko::Executor> exec,
        gko::size_type iteration_count)
    {
        return solver_type::build()
            .with_criteria(gko::stop::Iteration::build()
                               .with_max_iters(iteration_count)
                               .on(exec))
            .with_krylov_dim(2u);
    }

    static typename solver_type::parameters_type build_preconditioned(
        std::shared_ptr<const gko::Executor> exec,
        gko::size_type iteration_count)
    {
        return solver_type::build()
            .with_criteria(gko::stop::Iteration::build()
                               .with_max_iters(iteration_count)
                               .on(exec))
            .with_preconditioner(
                gko::preconditioner::Jacobi<value_type, index_type>::build()
                    .with_max_block_size(1u)
                    .on(exec))
            .with_krylov_dim(2u);
    }
};


struct CbGmres10 : SimpleSolverTest<gko::solver::CbGmres<solver_value_type>> {
    static typename solver_type::parameters_type build(
        std::shared_ptr<const gko::Executor> exec,
        gko::size_type iteration_count)
    {
        return solver_type::build()
            .with_criteria(gko::stop::Iteration::build()
                               .with_max_iters(iteration_count)
                               .on(exec))
            .with_krylov_dim(10u);
    }

    static typename solver_type::parameters_type build_preconditioned(
        std::shared_ptr<const gko::Executor> exec,
        gko::size_type iteration_count)
    {
        return solver_type::build()
            .with_criteria(gko::stop::Iteration::build()
                               .with_max_iters(iteration_count)
                               .on(exec))
            .with_preconditioner(
                gko::preconditioner::Jacobi<value_type, index_type>::build()
                    .with_max_block_size(1u)
                    .on(exec))
            .with_krylov_dim(10u);
    }
};


struct Gmres2 : SimpleSolverTest<gko::solver::Gmres<solver_value_type>> {
    static typename solver_type::parameters_type build(
        std::shared_ptr<const gko::Executor> exec,
        gko::size_type iteration_count)
    {
        return solver_type::build()
            .with_criteria(gko::stop::Iteration::build()
                               .with_max_iters(iteration_count)
                               .on(exec))
            .with_krylov_dim(2u);
    }

    static typename solver_type::parameters_type build_preconditioned(
        std::shared_ptr<const gko::Executor> exec,
        gko::size_type iteration_count)
    {
        return solver_type::build()
            .with_criteria(gko::stop::Iteration::build()
                               .with_max_iters(iteration_count)
                               .on(exec))
            .with_preconditioner(
                gko::preconditioner::Jacobi<value_type, index_type>::build()
                    .with_max_block_size(1u)
                    .on(exec))
            .with_krylov_dim(2u);
    }
};


struct Gmres10 : SimpleSolverTest<gko::solver::Gmres<solver_value_type>> {
    static typename solver_type::parameters_type build(
        std::shared_ptr<const gko::Executor> exec,
        gko::size_type iteration_count)
    {
        return solver_type::build()
            .with_criteria(gko::stop::Iteration::build()
                               .with_max_iters(iteration_count)
                               .on(exec))
            .with_krylov_dim(10u);
    }

    static typename solver_type::parameters_type build_preconditioned(
        std::shared_ptr<const gko::Executor> exec,
        gko::size_type iteration_count)
    {
        return solver_type::build()
            .with_criteria(gko::stop::Iteration::build()
                               .with_max_iters(iteration_count)
                               .on(exec))
            .with_preconditioner(
                gko::preconditioner::Jacobi<value_type, index_type>::build()
                    .with_max_block_size(1u)
                    .on(exec))
            .with_krylov_dim(10u);
    }
};


struct LowerTrs : SimpleSolverTest<gko::solver::LowerTrs<solver_value_type>> {
    static typename solver_type::parameters_type build_preconditioned(
        std::shared_ptr<const gko::Executor>, gko::size_type);
};


struct UpperTrs : SimpleSolverTest<gko::solver::UpperTrs<solver_value_type>> {
    static typename solver_type::parameters_type build_preconditioned(
        std::shared_ptr<const gko::Executor>, gko::size_type);
};


template <typename ObjectType>
struct test_pair {
    std::shared_ptr<ObjectType> ref;
    std::shared_ptr<ObjectType> dev;

    test_pair(std::unique_ptr<ObjectType> ref_obj,
              std::shared_ptr<const gko::Executor> exec)
        : ref{std::move(ref_obj)}, dev{gko::clone(exec, ref)}
    {}

    test_pair(std::unique_ptr<ObjectType> ref_obj,
              std::unique_ptr<ObjectType> dev_obj)
        : ref{std::move(ref_obj)}, dev{std::move(dev_obj)}
    {}
};


template <typename T>
class Solver : public ::testing::Test {
protected:
    using Config = T;
    using SolverType = typename T::solver_type;
    using Mtx = typename T::matrix_type;
    using index_type = gko::int32;
    using value_type = typename Mtx::value_type;
    using mixed_value_type = gko::next_precision<value_type>;
    using Vec = gko::matrix::Dense<value_type>;
    using MixedVec = gko::matrix::Dense<mixed_value_type>;

    Solver() : rand_engine(15) {}

    void SetUp()
    {
        ref = gko::ReferenceExecutor::create();
        init_executor(ref, exec);
    }

    void TearDown()
    {
        if (exec != nullptr) {
            ASSERT_NO_THROW(exec->synchronize());
        }
    }

    test_pair<Mtx> gen_mtx(int num_rows, int num_cols, int min_cols,
                           int max_cols)
    {
        auto mtx = gko::test::generate_random_matrix<Mtx>(
            num_rows, num_cols,
            std::uniform_int_distribution<>(min_cols, max_cols),
            std::normal_distribution<>(0.0, 1.0), rand_engine, ref);
        // make sure the matrix is well-conditioned
        gko::test::make_hpd(mtx.get(), 1.2);
        return test_pair<Mtx>{std::move(mtx), exec};
    }

    template <typename ValueType, typename IndexType>
    gko::matrix_data<ValueType, IndexType> gen_dense_data(gko::dim<2> size)
    {
        return {
            size,
            std::normal_distribution<gko::remove_complex<ValueType>>(0.0, 1.0),
            rand_engine};
    }

    template <typename VecType = Vec>
    test_pair<VecType> gen_in_vec(const test_pair<SolverType>& mtx, int nrhs,
                                  int stride)
    {
        auto size = gko::dim<2>{mtx.ref->get_size()[1],
                                static_cast<gko::size_type>(nrhs)};
        auto result = VecType::create(ref, size, stride);
        result->read(gen_dense_data<typename VecType::value_type,
                                    typename Mtx::index_type>(size));
        return {std::move(result), exec};
    }

    template <typename VecType = Vec>
    test_pair<VecType> gen_scalar()
    {
        return {gko::initialize<VecType>(
                    {gko::test::detail::get_rand_value<
                        typename VecType::value_type>(
                        std::normal_distribution<
                            gko::remove_complex<typename VecType::value_type>>(
                            0.0, 1.0),
                        rand_engine)},
                    ref),
                exec};
    }

    template <typename VecType = Vec>
    test_pair<VecType> gen_out_vec(const test_pair<SolverType>& mtx, int nrhs,
                                   int stride)
    {
        auto size = gko::dim<2>{mtx.ref->get_size()[0],
                                static_cast<gko::size_type>(nrhs)};
        auto result = VecType::create(ref, size, stride);
        result->read(gen_dense_data<typename VecType::value_type,
                                    typename Mtx::index_type>(size));
        return {std::move(result), exec};
    }

    double tol() { return 1e12 * r<value_type>::value; }

    double mixed_tol() { return 1e6 * r_mixed<value_type, mixed_value_type>(); }

    template <typename TestFunction>
    void forall_matrix_scenarios(TestFunction fn)
    {
        auto guarded_fn = [&](auto mtx) {
            try {
                fn(std::move(mtx));
            } catch (std::exception& e) {
                FAIL() << e.what();
            }
        };
        {
            SCOPED_TRACE("Empty matrix (0x0)");
            guarded_fn(gen_mtx(0, 0, 0, 0));
        }
        {
            SCOPED_TRACE("Sparse Matrix with variable row nnz (50x50)");
            guarded_fn(gen_mtx(50, 50, 10, 50));
        }
    }

    template <typename TestFunction>
    void forall_solver_scenarios(const test_pair<Mtx>& mtx, TestFunction fn)
    {
        auto guarded_fn = [&](auto solver) {
            try {
                fn(std::move(solver));
            } catch (std::exception& e) {
                FAIL() << e.what();
            }
        };
        {
            SCOPED_TRACE("Unpreconditioned solver with 0 iterations");
            guarded_fn(test_pair<SolverType>{
                Config::build(ref, 0).on(ref)->generate(mtx.ref),
                Config::build(exec, 0).on(exec)->generate(mtx.dev)});
        }
        {
            SCOPED_TRACE("Preconditioned solver with 0 iterations");
            guarded_fn(test_pair<SolverType>{
                Config::build_preconditioned(ref, 0).on(ref)->generate(mtx.ref),
                Config::build_preconditioned(exec, 0).on(exec)->generate(
                    mtx.dev)});
        }
        {
            SCOPED_TRACE("Unpreconditioned solver with 4 iterations");
            guarded_fn(test_pair<SolverType>{
                Config::build(ref, 4).on(ref)->generate(mtx.ref),
                Config::build(exec, 4).on(exec)->generate(mtx.dev)});
        }
        {
            SCOPED_TRACE("Preconditioned solver with 4 iterations");
            guarded_fn(test_pair<SolverType>{
                Config::build_preconditioned(ref, 4).on(ref)->generate(mtx.ref),
                Config::build_preconditioned(exec, 4).on(exec)->generate(
                    mtx.dev)});
        }
    }

    template <typename VecType = Vec, typename TestFunction>
    void forall_vector_scenarios(const test_pair<SolverType>& solver,
                                 TestFunction fn)
    {
        auto guarded_fn = [&](auto b, auto x) {
            try {
                fn(std::move(b), std::move(x));
            } catch (std::exception& e) {
                FAIL() << e.what();
            }
        };
        {
            SCOPED_TRACE("Multivector with 0 columns");
            guarded_fn(gen_in_vec<VecType>(solver, 0, 0),
                       gen_out_vec<VecType>(solver, 0, 0));
        }
        {
            SCOPED_TRACE("Single vector");
            guarded_fn(gen_in_vec<VecType>(solver, 1, 1),
                       gen_out_vec<VecType>(solver, 1, 1));
        }
        {
            SCOPED_TRACE("Single strided vector");
            guarded_fn(gen_in_vec<VecType>(solver, 1, 2),
                       gen_out_vec<VecType>(solver, 1, 3));
        }
        if (!gko::is_complex<value_type>()) {
            // check application of real matrix to complex vector
            // viewed as interleaved real/imag vector
            using complex_vec = gko::to_complex<VecType>;
            {
                SCOPED_TRACE("Single strided complex vector");
                guarded_fn(gen_in_vec<complex_vec>(solver, 1, 2),
                           gen_out_vec<complex_vec>(solver, 1, 3));
            }
            {
                SCOPED_TRACE("Strided complex multivector with 2 columns");
                guarded_fn(gen_in_vec<complex_vec>(solver, 2, 3),
                           gen_out_vec<complex_vec>(solver, 2, 4));
            }
        }
        {
            SCOPED_TRACE("Multivector with 2 columns");
            guarded_fn(gen_in_vec<VecType>(solver, 2, 2),
                       gen_out_vec<VecType>(solver, 2, 2));
        }
        {
            SCOPED_TRACE("Strided multivector with 2 columns");
            guarded_fn(gen_in_vec<VecType>(solver, 2, 3),
                       gen_out_vec<VecType>(solver, 2, 4));
        }
        {
            SCOPED_TRACE("Multivector with 40 columns");
            guarded_fn(gen_in_vec<VecType>(solver, 40, 40),
                       gen_out_vec<VecType>(solver, 40, 40));
        }
        {
            SCOPED_TRACE("Strided multivector with 40 columns");
            guarded_fn(gen_in_vec<VecType>(solver, 40, 43),
                       gen_out_vec<VecType>(solver, 40, 45));
        }
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::EXEC_TYPE> exec;

    std::default_random_engine rand_engine;
};

using SolverTypes =
    ::testing::Types<Cg, Cgs, Fcg, Bicg, Bicgstab, Idr1, Idr4, Ir, CbGmres2,
                     CbGmres10, Gmres2, Gmres10 /*, LowerTrs, UpperTrs*/>;

TYPED_TEST_SUITE(Solver, SolverTypes, TypenameNameGenerator);


TYPED_TEST(Solver, ApplyIsEquivalentToRef)
{
    this->forall_matrix_scenarios([&](auto mtx) {
        this->forall_solver_scenarios(mtx, [&](auto solver) {
            this->forall_vector_scenarios(solver, [&](auto b, auto x) {
                solver.ref->apply(b.ref.get(), x.ref.get());
                solver.dev->apply(b.dev.get(), x.dev.get());

                GKO_ASSERT_MTX_NEAR(x.ref, x.dev, this->tol());
            });
        });
    });
}


TYPED_TEST(Solver, AdvancedApplyIsEquivalentToRef)
{
    this->forall_matrix_scenarios([&](auto mtx) {
        this->forall_solver_scenarios(mtx, [&](auto solver) {
            this->forall_vector_scenarios(solver, [&](auto b, auto x) {
                auto alpha = this->gen_scalar();
                auto beta = this->gen_scalar();

                solver.ref->apply(alpha.ref.get(), b.ref.get(), alpha.ref.get(),
                                  x.ref.get());
                solver.dev->apply(alpha.dev.get(), b.dev.get(), alpha.dev.get(),
                                  x.dev.get());

                GKO_ASSERT_MTX_NEAR(x.ref, x.dev, this->tol());
            });
        });
    });
}


#if !(GINKGO_DPCPP_SINGLE_MODE)
TYPED_TEST(Solver, MixedApplyIsEquivalentToRef)
{
    using MixedVec = typename TestFixture::MixedVec;
    this->forall_matrix_scenarios([&](auto mtx) {
        this->forall_solver_scenarios(mtx, [&](auto solver) {
            this->template forall_vector_scenarios<MixedVec>(
                solver, [&](auto b, auto x) {
                    solver.ref->apply(b.ref.get(), x.ref.get());
                    solver.dev->apply(b.dev.get(), x.dev.get());

                    GKO_ASSERT_MTX_NEAR(x.ref, x.dev, this->mixed_tol());
                });
        });
    });
}


TYPED_TEST(Solver, MixedAdvancedApplyIsEquivalentToRef)
{
    using MixedVec = typename TestFixture::MixedVec;
    this->forall_matrix_scenarios([&](auto mtx) {
        this->forall_solver_scenarios(mtx, [&](auto solver) {
            this->template forall_vector_scenarios<MixedVec>(
                solver, [&](auto b, auto x) {
                    auto alpha = this->template gen_scalar<MixedVec>();
                    auto beta = this->template gen_scalar<MixedVec>();

                    solver.ref->apply(alpha.ref.get(), b.ref.get(),
                                      alpha.ref.get(), x.ref.get());
                    solver.dev->apply(alpha.dev.get(), b.dev.get(),
                                      alpha.dev.get(), x.dev.get());

                    GKO_ASSERT_MTX_NEAR(x.ref, x.dev, this->mixed_tol());
                });
        });
    });
}
#endif
