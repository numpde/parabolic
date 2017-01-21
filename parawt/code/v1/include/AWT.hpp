//
//  AWT.hpp
//  round1
//
//  Created by r. on 10/05/14.
//  Modeled on code by R. Andreev, 2013.12.06
//  Copyright (c) 2014 RICAM. All rights reserved.
//

#ifndef round1_AWT_hpp
#define round1_AWT_hpp

#include <boost/numeric/ublas/vector_of_vector.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/vector.hpp>

//#include <algorithm>
#include <numeric>
#include <vector>

#include "../include/stopwatch.hpp"

namespace spacetime
{
	namespace ublas = boost::numeric::ublas;
	
	struct AWT
	{
	public:
		typedef boost::numeric::ublas::compressed_matrix<double> matrix;
		typedef boost::numeric::ublas::vector<double> vector;
	public:
		class Filter
		{
		public:
			const bool wave;
			const unsigned int n, N;
			const matrix C, F;
		public:
			Filter(unsigned int n, unsigned int N, const matrix& C, const matrix& F)
			:
			wave(n!=0), n(n), N(N), C(C), F(F)
			{
				assert(C.size1() == n);
				assert(C.size2() == N);
				assert(F.size1() == N-n);
				assert(F.size2() == N);
			}
		};
	public:
		// Should be hidden
		std::vector<Filter> filterbank;
	public:
		// These should be hidden
		matrix V;
		vector gamma;
		unsigned int n;
		unsigned int nu;
		double eta;
	public:
		AWT() : n(0), V(0,0), gamma(0), nu(0), eta(0) { }
	public:
		AWT(const matrix& A, const matrix& M, unsigned int nu = 2, double eta = 1.9)
		:
		V(0, 0), gamma(0), n((unsigned int)(A.size1())), nu(nu), eta(eta), filterbank()
		// Pre: A and M are square matrices of equal size
		{
			stopwatch::Time time(stopwatch::watches, "spacetime::AWT::AWT()");
			
			assert((A.size1() == n) && (A.size2() == n));
			assert((M.size1() == n) && (M.size2() == n));
			assert(eta > 0);
			
			// Will hold the square roots of generalized "eigenvalues"
			gamma = vector(n);
			gamma.clear();
			
			typedef ublas::compressed_matrix<double, ublas::row_major> S_type;
			typedef ublas::compressed_matrix<double, ublas::column_major> T_type;
			
			// Note:
			// prod(S_type, T_type) is more efficient than prod(T_type, S_type)
			
			T_type T = ublas::identity_matrix<>(n);
			T_type PRO = ublas::identity_matrix<>(n);
			{
				typedef ublas::compressed_matrix<double> acmc_type;
				acmc_type mc = M, ac = A;
				unsigned int m = n;
				// Invariant: m is size1 and size2 of mc and ac
				
				// Iterate over levels
				while (m >= 3)
				{
					const unsigned int m0 = m;
					S_type M = mc;
					S_type A = ac;
					
					// Step 1. Identify fine hats
					
					typedef ublas::compressed_vector<double> compressed_vector;
					typedef std::vector<compressed_vector> P_type;
					P_type P;
					{
						//stopwatch::Time time(stopwatch::watches, "spacetime::AWT::AWT().P");
						
						for (unsigned int i = 0; i != m0; ++i) {
							P.push_back(compressed_vector(m0));
							(P.back())[i] = 1;
						}
					}
					
					std::vector<unsigned int> I(m0);
					{
						for (unsigned int i = 0; i != m0; ++i) I[i] = i;
					}
					
					double e0 = 0;
					while (m >= 3)
					{
						unsigned int j; // index of most energetic hat
						double e1; // and its energy
						{
							//stopwatch::Time time(stopwatch::watches, "spacetime::AWT::AWT().e1");
							std::vector<double> E1(m);
							{
								ublas::range ALL(0, m0);
								ublas::matrix_vector_range<acmc_type> dac(ac, ALL, ALL);
								ublas::matrix_vector_range<acmc_type> dmc(mc, ALL, ALL);
								for (unsigned i = 0; i != m; ++i)
									E1[i] = dac(I[i]) / dmc(I[i]);
								E1.front() = E1.back() = 0;
							}
							auto i = std::max_element(E1.begin(), E1.end());
							e1 = *i;
							j = (unsigned int)(std::distance(E1.begin(), i));
							assert(E1[j] == e1);
						}
						assert(e1 > 0);
						
						if (e1 <= e0 / eta) break;
						if (e0 == 0) e0 = e1;
						
						double p = -ac(I[j-1], I[j]) / ac(I[j], I[j]);
						double q = -ac(I[j+1], I[j]) / ac(I[j], I[j]);
						
						// Manipulate ac
						{
							//stopwatch::Time time(stopwatch::watches, "spacetime::AWT::AWT().prod1");
							
							row(ac, I[j-1]) += p * row(ac, I[j]);
							row(ac, I[j+1]) += q * row(ac, I[j]);
							
							I.erase(I.begin() + j);
							
							// New effective size of mc and ac
							m--;
						}
						
						// Pre-Manipulate T by means of P
						{
							//stopwatch::Time time(stopwatch::watches, "spacetime::AWT::AWT().mani");
							P[j-1] += p * P[j];
							P[j+1] += q * P[j];
							P.push_back(P[j]);
							P.erase(P.begin() + j);
						}
					} // Finished identifying level
					
					S_type SC(m, m0), SF(m0-m, m0);
					{
						//stopwatch::Time time(stopwatch::watches, "spacetime::AWT::AWT().perm");
						
						for (unsigned int i = 0; i != m; ++i) row(SC, i) = P[i];
						for (unsigned int i = m; i != m0; ++i) row(SF, i-m) = P[i];
					}
					
					T_type MSC;
					{
						//stopwatch::Time time(stopwatch::watches, "spacetime::AWT::AWT().mcac");
						T_type tSC(ublas::trans(SC));
						MSC = prod(M, tSC);
						mc = prod(SC, MSC);
						ac = prod(SC, T_type(prod(A, tSC)));
					}
					
					// Step 2. Orthogonalize level
					{
						T_type DSC(SC);
						{
							//stopwatch::Time time(stopwatch::watches, "spacetime::AWT::AWT().mass");
							
							for (unsigned int i = 0; i != m; ++i)
							{
								ublas::matrix_row<acmc_type> r(mc, i);
								row(DSC, i) *= (1. / std::accumulate(r.begin(), r.end(), 0.));
							}
						}
						
						ublas::range IF(m, m0), ALL(0, m0);
						for (unsigned int k = 0; k != nu; ++k)
						{
							//stopwatch::Time time(stopwatch::watches, "spacetime::AWT::AWT().proj2");
							assert(SF.size2() == MSC.size1());
							assert(MSC.size2() == DSC.size1());
							assert(SF.size2() == DSC.size2());
							SF -= prod(S_type(prod(SF, MSC)), DSC);
						}
					}
					
					// Normalize wavelets
					{
						//stopwatch::Time time(stopwatch::watches, "spacetime::AWT::AWT().norm");
						for (unsigned int i = 0; i != m0-m; ++i)
						{
							auto c = row(SF, i);
							c *= (1. / sqrt(ublas::inner_prod(c, prod(M, ublas::trans(c)))));
							double d = ublas::inner_prod(c, prod(A, ublas::trans(c)));
							gamma[i + m] = sqrt(std::max(0., d));
						}
					}
					
					// Update filterbank
					{
						Filter filter(m, m0, SC, SF);
						filterbank.push_back(filter);
					}
					
					// Copy S into T
					{
						//stopwatch::Time time(stopwatch::watches, "spacetime::AWT::AWT().copy1");
						noalias(project(T, ublas::range(m, m0), ublas::range(0, n))) = prod(SF, PRO);
					}
					
					// Update prolongation onto finest hats
					{
						//stopwatch::Time time(stopwatch::watches, "spacetime::AWT::AWT().PRO2");
						PRO = prod(SC, PRO);
					}

				} // end while(m >= 3)
				
				{
					S_type SF = ublas::identity_matrix<>(m);;
					{
						for (unsigned int i = 0; i != m; ++i)
						{
							auto c = row(SF, i);
							c *= (1. / sqrt(mc(i, i)));
							double d = ublas::inner_prod(c, prod(ac, ublas::trans(c)));
							gamma[i] = sqrt(std::max(0., d));
						}
					}
					
					// Update filterbank
					{
						Filter filter(0, m, matrix(0, m), SF);
						filterbank.push_back(filter);
					}
					
					//stopwatch::Time time(stopwatch::watches, "spacetime::AWT::AWT().copy2");
					noalias(project(T, ublas::range(0, m), ublas::range(0, n))) = prod(SF, PRO);
				}
			} // Matrix T computed
			
			// Compute V
			{
				//stopwatch::Time time(stopwatch::watches, "spacetime::AWT::AWT().V");
				
				V = ublas::trans(T);
			}
		}
		
		AWT& operator=(const AWT&) = default;
	};

}

#endif
