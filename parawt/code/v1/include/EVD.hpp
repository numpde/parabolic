//
//  EVD.hpp
//  round1
//
//  Created by r. on 08/05/14.
//  Copyright (c) 2014 RICAM. All rights reserved.
//

#ifndef round1_EVD_hpp
#define round1_EVD_hpp

// GSL
#include <gsl/gsl_eigen.h>

// Boost
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>

//#include <boost/numeric/ublas/vector_sparse.hpp>
//#include <boost/numeric/ublas/vector_proxy.hpp>
//#include <boost/numeric/ublas/matrix_sparse.hpp>
//#include <boost/numeric/ublas/operation.hpp>
//#include <boost/numeric/ublas/operation_sparse.hpp>

namespace gsl
{
	struct EVD
	{
	public:
		typedef boost::numeric::ublas::matrix<double> matrix;
		typedef boost::numeric::ublas::vector<double> vector;
	public:
		// These should not be changed from outside
		unsigned int n;
		matrix V;
		vector d;
	public:
		EVD() : n(0), V(0,0), d(0) { }
	private:
		/*//
		 void testV(const matrix& AE, const matrix& ME) const
		 {
		 namespace ublas = boost::numeric::ublas;
		 
		 vector u(n);
		 for (unsigned int i = 0; i != u.size(); ++i) u(i) = i;
		 double j = 1, k = 2;
		 
		 cout << "A: " << AE << endl;
		 cout << "M: " << ME << endl;
		 cout << "V: " << V << endl;
		 cout << "d: " << d << endl;
		 matrix W = ublas::trans(V);
		 matrix AV = ublas::prod(AE, V), MV = ublas::prod(ME, V);
		 cout << "Vt A V: " << ublas::prod(W, AV) << endl;
		 cout << "Vt M V: " << ublas::prod(W, MV) << endl;
		 
		 vector f1(n);
		 {
		 f1 = ublas::prod(j*AE + k*ME, u);
		 }
		 vector f2(n);
		 {
		 f2 = ublas::prod(ME, u);
		 f2 = ublas::prod(ublas::trans(V), f2);
		 for (unsigned int i = 0; i != f2.size(); ++i)
		 {
		 f2(i) *= (j * d(i) + k);
		 }
		 f2 = ublas::prod(V, f2);
		 f2 = ublas::prod(ME, f2);
		 }
		 cout << f1 << endl;
		 cout << f2 << endl;
		 cout << "testV residual: " << ublas::norm_inf(f1 - f2) << endl;
		 assert(ublas::norm_inf(f1 - f2) <= 1e-6);
		 }
		 //*/
		
	public:
		EVD(const matrix& AE, const matrix& ME)
		:
		n(0), V(0,0), d(0)
		// Pre: AE and ME are square matrices of equal size
		{
			// Thanks to:
			// http://sector7.xray.aps.anl.gov/~dohnarms/programming/gsl/gsl-ref.pdf
			// http://www.ryolab.com/soft/aper/dd/d99/_2utility_2algorithm_2eigen_2eigentest_2main_8cpp-example.html
			
			n = (unsigned int)(AE.size1());
			
			assert((AE.size1() == n) && (AE.size2() == n));
			assert((ME.size1() == n) && (ME.size2() == n));
			
			gsl_eigen_gensymmv_workspace * w = gsl_eigen_gensymmv_alloc(n);
			
			matrix A(n, n); A = AE;
			matrix M(n, n); M = ME;
			gsl_matrix_view a = gsl_matrix_view_array(&A.data()[0], n, n);
			gsl_matrix_view m = gsl_matrix_view_array(&M.data()[0], n, n);
			gsl_matrix * U = gsl_matrix_alloc(n , n);
			gsl_vector * c = gsl_vector_alloc(n);
			gsl_eigen_gensymmv(&a.matrix, &m.matrix, c, U, w);
			gsl_eigen_gensymmv_sort(c, U, GSL_EIGEN_SORT_ABS_ASC);
			
			V = matrix(n, n);
			d = vector(n);
			std::copy(U->data, U->data + (n*n), &(V.data()[0]));
			std::copy(c->data, c->data + n, &(d.data()[0]));
			
			gsl_vector_free(c);
			gsl_matrix_free(U);
			gsl_eigen_gensymmv_free(w);
			
			// M-normalize V
			for (unsigned int i = 0; i != n; ++i)
			{
				namespace ublas = boost::numeric::ublas;
				auto c = ublas::column(V, i);
				double norm = ublas::inner_prod(c, ublas::prod(ME, c));
				//cout << "norm: " << norm << endl;
				c *= (1/sqrt(norm));
			}
			
			//testV(AE, ME);
		}
		
		EVD& operator=(const EVD&) = default;
	};
} // namespace tools

#endif
