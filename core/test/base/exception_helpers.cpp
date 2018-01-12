/*
 * Copyright 2017-2018
 *
 * Karlsruhe Institute of Technology
 *
 * Universitat Jaume I
 *
 * University of Tennessee
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <core/base/exception_helpers.hpp>


#include <gtest/gtest.h>


namespace {


void not_implemented_func() NOT_IMPLEMENTED;


TEST(NotImplemented, ThrowsWhenUsed)
{
    ASSERT_THROW(not_implemented_func(), gko::NotImplemented);
}


void not_compiled_func() NOT_COMPILED(cpu);


TEST(NotCompiled, ThrowsWhenUsed)
{
    ASSERT_THROW(not_compiled_func(), gko::NotCompiled);
}


void does_not_support_int() { throw NOT_SUPPORTED(int); }


TEST(NotSupported, ReturnsNotSupportedException)
{
    ASSERT_THROW(does_not_support_int(), gko::NotSupported);
}


TEST(AssertConformant, DoesNotThrowWhenConformant)
{
    ASSERT_NO_THROW(ASSERT_CONFORMANT(gko::size(3, 5), gko::size(5, 6)));
}


TEST(AssertConformant, ThrowsWhenNotConformant)
{
    ASSERT_THROW(ASSERT_CONFORMANT(gko::size(3, 5), gko::size(7, 3)),
                 gko::DimensionMismatch);
}


TEST(AssertEqualRows, DoesNotThrowWhenEqualRowSize)
{
    ASSERT_NO_THROW(ASSERT_EQUAL_ROWS(gko::size(5, 3), gko::size(5, 6)));
}


TEST(AssertEqualRows, ThrowsWhenDifferentRowSize)
{
    ASSERT_THROW(ASSERT_EQUAL_ROWS(gko::size(3, 5), gko::size(7, 3)),
                 gko::DimensionMismatch);
}


TEST(AssertEqualCols, DoesNotThrowWhenEqualColSize)
{
    ASSERT_NO_THROW(ASSERT_EQUAL_COLS(gko::size(3, 6), gko::size(5, 6)));
}


TEST(AssertEqualCols, ThrowsWhenDifferentColSize)
{
    ASSERT_THROW(ASSERT_EQUAL_COLS(gko::size(3, 5), gko::size(7, 3)),
                 gko::DimensionMismatch);
}


TEST(AssertEqualDimensions, DoesNotThrowWhenEqualDimensions)
{
    ASSERT_NO_THROW(ASSERT_EQUAL_DIMENSIONS(gko::size(5, 6), gko::size(5, 6)));
}


TEST(AssertEqualDimensions, ThrowsWhenDifferentDimensions)
{
    ASSERT_THROW(ASSERT_EQUAL_DIMENSIONS(gko::size(3, 5), gko::size(7, 5)),
                 gko::DimensionMismatch);
}


TEST(EnsureAllocated, DoesNotThrowWhenAllocated)
{
    int x = 5;
    ASSERT_NO_THROW(ENSURE_ALLOCATED(&x, "CPU", 4));
}


TEST(EnsureAllocated, ThrowsWhenNotAllocated)
{
    ASSERT_THROW(ENSURE_ALLOCATED(nullptr, "CPU", 20), gko::AllocationError);
}


}  // namespace
