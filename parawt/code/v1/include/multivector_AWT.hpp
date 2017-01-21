//
//  multivector_AWT.hpp
//  round1
//
//  R. Andreev, 2014-2015
//

#ifndef round1_multivector_AWT_hpp
#define round1_multivector_AWT_hpp


#include <boost/numeric/ublas/vector_of_vector.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/vector.hpp>

//#include <algorithm>
#include <numeric>
#include <vector>

#include "stopwatch.hpp"
#include "multivector.hpp"

namespace mypara
{
	namespace ublas = boost::numeric::ublas;
	
	struct AWT
	{
	public:
		typedef ublas::compressed_matrix<double> matrix;
		typedef ublas::vector<double> vector;
	public:
		class Filter
		{
		public:
			typedef ublas::compressed_matrix<double, ublas::column_major> sparse_matrix;
		public:
			const bool wave;
			const unsigned int n, N;
			const sparse_matrix C, F;
			const sparse_matrix Ct, Ft;
			mypara::Jplan jplan_C, jplan_Ct;
			mypara::Jplan jplan_F, jplan_Ft;
		public:
			Filter(unsigned int n, unsigned int N, const sparse_matrix& C, const sparse_matrix& F)
			:
			wave(n!=0), n(n), N(N),
			C(C), F(F),
			Ct(ublas::trans(C)), Ft(ublas::trans(F)),
			jplan_C(), jplan_Ct(),
			jplan_F(), jplan_Ft()
			{
				assert(C.size1() == n);
				assert(C.size2() == N);
				assert(F.size1() == N-n);
				assert(F.size2() == N);
			}
		};
	private:
		// Should be hidden
		typedef std::vector<Filter> FilterBank;
		FilterBank filterbank;
	public:
		// These should be hidden
		vector gamma;
		unsigned int n;
		unsigned int nu;
		double eta;
	public:
		AWT() : n(0), gamma(0), nu(0), eta(0), filterbank() { }
	public:
		AWT(const matrix& A, const matrix& M, unsigned int nu = 1, double eta = 1.9)
		:
		gamma(0), n((unsigned int)(A.size1())), nu(nu), eta(eta), filterbank()
		// Pre: A and M are square matrices of equal size
		{
			stopwatch::Time time(stopwatch::watches, "mypara::AWT::AWT");
			
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
			} // end while(m >= 3)
			
			{
				// Normalize ground hat functions
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
			}
		}
	public:
		AWT& operator=(const AWT&) = default;
	public:
		multivector uT(const multivector& u) { return this->uT(u, filterbank.begin()); }
		multivector uTt(const multivector& u) { return this->uTt(u, filterbank.begin()); }
	private:
		multivector uT(const multivector& u, const FilterBank::iterator i);
		multivector uTt(const multivector& u, const FilterBank::iterator i);
	};
	
	//
	
	multivector
	AWT::uT(const multivector& u, const AWT::FilterBank::iterator i)
	// Note (RA, 20160509):
	// This routine is called uT because
	// morally it computes "T * column vector",
	// (but here we have row vectors by
	//  the convention of putting temporal
	//  degrees of freedom along the j-dimension).
	// This is why we have here Ft and Ct (not F and C).
	{
		stopwatch::Time time(stopwatch::watches, "mypara::AWT::uT");
		
		if ((i->n) == 0)
		{
            multivector v = u;
//            mypara::prod(u, ublas::trans(i->F), i->jplan_Ft, v);
            mypara::prod(u, i->Ft, v);
			return v;
		}
		
		multivector uc(u.getcm()), uf(u.getcm());
        u.split(uc, uf, i->n);
		
		// mypara::prod(u, ublas::trans(i->F), i->jplan_Ft(mypara::Jplan::Strategy::jreverse), uf);
        mypara::prod(u, i->Ft, uf);
		
		// mypara::prod(u, ublas::trans(i->C), i->jplan_Ct, uc);
        mypara::prod(u, i->Ct, uc);
		
        if (uc.shrink_communicator()) {
			uc = AWT::uT(uc, i + 1);
        }
		
		multivector v = u; v.clear();
        v.merge(uc, uf);
		
		return v;
	}
	
	multivector
	AWT::uTt(const multivector& u, const AWT::FilterBank::iterator i)
	{
		stopwatch::Time time(stopwatch::watches, "mypara::AWT::uTt");
		
		if ((i->n) == 0)
		{
			multivector v = u;
			// mypara::prod(u, i->F, i->jplan_F, v);
            mypara::prod(u, i->F, v);
			return v;
		}
		
		multivector uc(u.getcm()), uf(u.getcm());
		{
			stopwatch::Time time(stopwatch::watches, "mypara::AWT::uTt.split");
			
			u.split(uc, uf, i->n);
		}
		
		multivector v2 = u; v2.clear();
		{
			stopwatch::Time time(stopwatch::watches, "mypara::AWT::uTt.prod1");
			
			// mypara::prod(uf, i->F, i->jplan_F, v2);
			mypara::prod(uf, i->F, v2);
		}

		mpi::communicator ucomm = uc.getcm();
		if (uc.shrink_communicator()) {
			stopwatch::Time time(stopwatch::watches, "mypara::AWT::uTt.rec");
			
            uc = uTt(uc, i + 1);
        }
        uc.embrace_communicator(ucomm);
		
		multivector v1 = u; v1.clear();
		{
			stopwatch::Time time(stopwatch::watches, "mypara::AWT::uTt.prod2");
			
			// mypara::prod(uc, i->C, i->jplan_C, v1);
			mypara::prod(uc, i->C, v1);
		}
		
		v1 += v2;
		
		return v1;
	}
}


#endif
