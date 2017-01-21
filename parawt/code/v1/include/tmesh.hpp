//
//  tmesh.hpp
//  round1
//
//  Created by r. on 09/05/14.
//  Copyright (c) 2014 RICAM. All rights reserved.
//

#ifndef round1_tmesh_hpp
#define round1_tmesh_hpp

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <vector>

namespace spacetime
{
	namespace ublas = boost::numeric::ublas;
	
	class tmesh : public ublas::vector<double>
	{
	public:
		typedef ublas::vector<double> parent_type;
	public:
		struct interval
		{
		public:
			// Should be hidden
			double a, b, len, mid;
		public:
			interval(double a, double b) : a(a), b(b), len(b-a), mid((a+b)/2) {}
		};
	public:
		void
		operator=(const tmesh& m) {
			parent_type::resize(m.size());
			parent_type::assign(m);
		}
	public:
		bool isvalid() const
		{
			const parent_type& x = (parent_type)(*this);
			if (x.empty()) return false;
			if (x.size() <= 1) return false;
			for (unsigned int i = 1; i != x.size(); ++i)
			{
				if (!(x(i-1) <= x(i))) return false;
			}
			return true;
		}
	public:
		unsigned int getni() const
		// Get the number of intervals
		{
			assert(isvalid());
			assert(size() >= 1);
			return (unsigned int)(size() - 1);
		}
	public:
		void makeuniform(double a, double b, unsigned int nodes)
		{
			assert(nodes >= 2);
			parent_type y(nodes);
			for (unsigned int n = 0; n != nodes; ++n)
				y[n] = a + n * ((b - a) / (nodes - 1));
			this->swap(y);
		}
		
		void refine()
		{
			assert(isvalid());
			parent_type vrefd(1 + 2 * getni());
			{
				const parent_type& x = (parent_type)(*this);
				unsigned int j = 0;
				vrefd(j++) = x(0);
				for (unsigned int i = 1; i != x.size(); ++i)
				{
					vrefd(j++) = 0.5 * (x(i-1) + x(i));
					vrefd(j++) = x(i);
				}
				assert(j == vrefd.size());
			}
			this->swap(vrefd);
		}
		
		interval getI(unsigned int n) const
		// n-th subinterval, n >= 1
		{
			assert(isvalid());
			const parent_type& x = (parent_type)(*this);
			assert((1 <= n) && (n < x.size()));
			return interval(x(n-1), x(n));
		}
		
		std::vector<interval> getIs() const
		// vector of all subintervals, canonically ordered
		{
			assert(isvalid());
			std::vector<interval> Is;
			{
				const parent_type& x = (parent_type)(*this);
				for (unsigned int n = 1; n != x.size(); ++n)
					Is.push_back(interval(x(n-1), x(n)));
			}
			return Is;
		}
	
		//namespace ublas = boost::numeric::ublas;
		typedef ublas::compressed_matrix<double, ublas::row_major> sparse_matrix;
		//
		sparse_matrix
		naive_prolongation_to() {
			const tmesh& m = (*this);
			// Number of elements
			unsigned K = (m.size() - 1);
			// ... should be even
			assert((K % 2) == 0);
			// Number of elements on coarse mesh
			unsigned k = K / 2;
			
			unsigned nf = K+1;
			unsigned nc = k+1;
			
			ublas::mapped_matrix<double> P(nf, nc);
			
			// Find coarse nodes
			for (unsigned j = 0; j != nc; ++j) {
				unsigned i = 2 * j;
				if (j != 0) P(i-1, j) = 0.5;
				P(i+0, j) = 1;
				if (j+1 != nc) P(i+1, j) = 0.5;
			}
			
			return P;
		}
	};
}


#endif
