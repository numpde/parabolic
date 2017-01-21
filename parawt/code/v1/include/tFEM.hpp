//
//  tFEM.hpp
//  round1
//
//  Created by r. on 09/05/14.
//  Copyright (c) 2014 RICAM. All rights reserved.
//

#ifndef round1_tFEM_hpp
#define round1_tFEM_hpp

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>

#include "tmesh.hpp"
#include "stopwatch.hpp"
//#include "EVD.hpp"
#include "AWT.hpp"

//#include <algorithm>

namespace spacetime
{
	namespace ublas = boost::numeric::ublas;

	class tFEM
	{
	public:
		typedef ublas::compressed_matrix<double> sparse_matrix;
		typedef ublas::matrix<double> dense_matrix;
		typedef ublas::vector<double> vector;
	public:
		// Should be hidden
		sparse_matrix MtE, AtE;
		sparse_matrix MtF;
		sparse_matrix CtFE, MtFE;
		sparse_matrix EtE;
		dense_matrix VtE;
		vector gamma;
//	public:
//		const sparse_matrix& getMtE() const { return MtE; }
//		const sparse_matrix& getAtE() const { return AtE; }
//		const sparse_matrix& getMtF() const { return MtF; }
//		const sparse_matrix& getCtFE() const { return CtFE; }
//		const sparse_matrix& getMtFE() const { return MtFE; }
//		const sparse_matrix& getEtE() const { return EtE; }
//		const dense_matrix& getVtE() const { return VtE; }
//		const vector& getgamma() const { return gamma; }
	private:
		tmesh TE, TF;
		unsigned int nref;
	public:
		// Should be hidden
		unsigned int dimE;
		unsigned int dimF;
	public:
		const tmesh& refTE() const { return TE; }
		const tmesh& refTF() const { return TF; }
//		unsigned int dimE() const { return (unsigned int)(TE.size()); }
//		unsigned int dimF() const { return (unsigned int)(TF.size() - 1); }
	public:
		void perform_evd()
		{
			stopwatch::Time time(stopwatch::watches, "tFEM::perform_evd()");
			
			{
				//gsl::EVD evd(AtE, MtE);
				//spacetime::AWT evd(AtE, MtE);
				//VtE = evd.V;
				//gamma = evd.d;
				//assert(gamma.size() == TE.size());
				
				VtE.resize(0, 0);
				gamma.resize(0);
			}
			
			for (vector::iterator i = gamma.begin(); i != gamma.end(); ++i)
				(*i) = sqrt(std::max(0.0, *i));
		}
	public:
		void assemble_matrices()
		{
			stopwatch::Time time(stopwatch::watches, "tFEM::assemble_matrices()");
			
			// nref >= 1 not implemented
			assert(nref == 0);
			
			typedef ublas::mapped_matrix<double> mapped_matrix;
			mapped_matrix MtE(dimE, dimE, 3*dimE); MtE.clear();
			mapped_matrix AtE(dimE, dimE, 3*dimE); AtE.clear();
			{
				std::vector<tmesh::interval> Is = TE.getIs();
				for (auto n = 0; n != Is.size(); ++n)
				{
					unsigned int m = n + 1;
					
					double h = Is[n].len;
					MtE(n, n) += (2./6.) * h;
					MtE(m, m) += (2./6.) * h;
					MtE(m, n) += (1./6.) * h;
					MtE(n, m) += (1./6.) * h;
					
					double g = 1 / h;
					AtE(n, n) += +g;
					AtE(m, m) += +g;
					AtE(m, n) += -g;
					AtE(n, m) += -g;
				}
			}
			
			mapped_matrix MtF(dimF, dimF, dimF); MtF.clear();
			{
				std::vector<tmesh::interval> Is = TF.getIs();
				for (auto i = 0; i != Is.size(); ++i)
				{
					MtF(i, i) = Is[i].len;
				}
			}
			
			mapped_matrix CtFE(dimF, dimE, 2*dimF); CtFE.clear();
			mapped_matrix MtFE(dimF, dimE, 2*dimF); MtFE.clear();
			{
				std::vector<tmesh::interval> Is = TE.getIs();
				for (auto n = 0; n != Is.size(); ++n)
				{
					MtFE(n, n+0) += (1./2.) * Is[n].len;
					MtFE(n, n+1) += (1./2.) * Is[n].len;
					
					CtFE(n, n+0) += -1;
					CtFE(n, n+1) += +1;
				}
			}
			
			mapped_matrix EtE(1, dimE, 1); EtE.clear();
			{
				EtE(0, 0) = 1;
			}
			
			this->MtE = MtE;
			this->AtE = AtE;
			this->MtF = MtF;
			this->CtFE = CtFE;
			this->MtFE = MtFE;
			this->EtE = EtE;
		}
	public:
		tFEM(const tmesh& te, unsigned nref = 0, bool perform_evd_now = true)
		:
		TE(te), TF(te), nref(nref), dimE(0), dimF(0)
		{
			for (unsigned int n = 0; n != nref; ++n)
				TF.refine();
			
			dimE = (unsigned int)(TE.size());
			dimF = (unsigned int)(TF.size() - 1);
			
			assemble_matrices();
			
			if (perform_evd_now) perform_evd();
		}
	};

} // namespace spacetime

#endif
