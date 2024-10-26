#include "mex.hpp"
#include "mexAdapter.hpp"
#include "sparse.hpp"
#include "dmumps_c.h"

class MexFunction 
    : public matlab::mex::Function 
{
    
public:
    MexFunction()  {
        matlabPtr = getEngine();
    }
    ~MexFunction() = default;
    void operator()(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs) {
        if (inputs.size() < 2)
            utilities::error("Two arguments are required. Provide A, b to return solution to A*x = b");
        matlab::data::ArrayFactory factory;
        utilities::Sparse<double> A;
        DMUMPS_STRUC_C id;

        if (utilities::issparse(inputs[0])){
            matlab::data::SparseArray<double> Amex = std::move(inputs[0]);
            A.set(Amex);
        } else {
            matlab::data::TypedArray<double> Amex = std::move(inputs[0]);
            A.set(Amex);
        }

        if (A.getNumberOfRows() != A.getNumberOfColumns())
            utilities::error("Matrix must be square");

        matlab::data::TypedArray<double> b = std::move(inputs[1]);

        if (A.getNumberOfRows() != b.getNumberOfElements())
            utilities::error("Matrix and vector dimensions must agree");

        constexpr MUMPS_INT JOB_INIT=-1, JOB_END=-2, JOB_SOLVE=6, USE_COMM_WORLD=-987654;

        id.comm_fortran=USE_COMM_WORLD;
        id.par=1; id.sym=0;
        id.job=JOB_INIT;
        dmumps_c(&id);

        id.icntl[1-1] = -1; // output stream for error messages
        id.icntl[2-1] = -1; // output stream for diagnostic printing
        id.icntl[3-1] = -1; // output stream for global information
        id.icntl[4-1] = 0; // print level
        id.icntl[5-1] = 0; // Elemental matrix format
        id.icntl[18-1] = 0; // no mpi
        
        id.n = A.getNumberOfRows(); 
        id.nnz = A.getNumberOfNonZeroElements(); 
        id.irn = new MUMPS_INT[A.getNumberOfNonZeroElements()];
        id.jcn = new MUMPS_INT[A.getNumberOfNonZeroElements()];
        id.a = new DMUMPS_COMPLEX[A.getNumberOfNonZeroElements()];
        id.rhs = new DMUMPS_COMPLEX[A.getNumberOfRows()];

        A.iRow(id.irn);
        A.jCol(id.jcn);
        A.val(id.a);

        std::transform(id.irn, id.irn + id.nnz, id.irn, [](MUMPS_INT i){return i+1;});
        std::transform(id.jcn, id.jcn + id.nnz, id.jcn, [](MUMPS_INT i){return i+1;});

        std::copy(b.begin(), b.end(), id.rhs);

        id.job=JOB_SOLVE;
        dmumps_c(&id);

        if (id.info[0] < 0)
            utilities::error("MUMPS failed to solve the system, error codes INFO(1): {}, INFO(2): {}", id.info[0], id.info[1]);

        matlab::data::TypedArray<double> x = factory.createArray<double>({id.n, 1});
        std::copy(id.rhs, id.rhs + id.n, x.begin());

        outputs[0] = std::move(x);

        id.job=JOB_END;
        dmumps_c(&id);
        delete[] id.irn;
        delete[] id.jcn;
        delete[] id.a;
        delete[] id.rhs;
    }
};