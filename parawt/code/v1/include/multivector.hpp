//
//  multivector.hpp
//  round1
//
//  Created by r. on 09/05/14.
//  Copyright (c) 2014 RICAM. All rights reserved.
//

#ifndef round1_multivector_hpp
#define round1_multivector_hpp

// Standard libraries
#include <algorithm>    // std::random_shuffle
#include <numeric> // std::iota
#include <iostream>
#include <vector>
#include <queue>
#include <string>
#include <sstream>
#include <list>
#include <set>
#include <limits>

// Boost MPI
#include <boost/mpi.hpp>
#include <boost/serialization/vector.hpp>

// Boost ublas
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/vector_sparse.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/io.hpp>

//
#include "../include/stopwatch.hpp"

namespace mypara
{
    
	namespace ublas = boost::numeric::ublas;
    namespace mpi = boost::mpi;
	
	class multivector : public ublas::matrix<double, ublas::column_major>
	{
	public:
		typedef ublas::matrix<double, ublas::column_major> parent_type;
		typedef parent_type::value_type value_type;
		typedef parent_type::size_type size_type;
		
		typedef ublas::matrix_column<parent_type> matrix_column;
		typedef ublas::matrix_column<const parent_type> const_matrix_column;
		
		typedef ublas::matrix<value_type, ublas::column_major> dense_matrix;
	private:
		// MPI
		mpi::communicator world;
	private:
		// Total number of verticals to distribute (horizontal size)
		unsigned int total;
		// For each process, begin and end of its portion
		std::vector<unsigned int> p2gja;
		std::vector<unsigned int> p2gjb;
	public:
		const mpi::communicator& getcm() const { return world; }
	public:
        // Width of the whole multivector
		unsigned int getot() const { return total; }
        // Rank in the communicator
		unsigned int getrk() const { return world.rank(); }
        // Begin of portion (global)
		unsigned int getja() const { return p2gja[getrk()]; }
        // End of portion (global)
		unsigned int getjb() const { return p2gjb[getrk()]; }
        // Size of portion
		unsigned int getsz() const { return (getjb() - getja()); }
	public:
		unsigned int gj2pr(unsigned int j) const
        // Finds the owner process of global column j
		{
			for (auto p = 0; p < world.size(); ++p)
				if ((p2gja[p] <= j) && (j < p2gjb[p]))
					return p;
			return world.size();
		}
		
		bool ismyj(unsigned int j) const { return ((getja() <= j) && (j < getjb())); }
	public:
		multivector(const mpi::communicator& comm)
		: world(comm), total(0), parent_type(0,0)
		{
		}
		
		multivector(size_type size1, unsigned int total, const mpi::communicator& comm)
		: world(comm), total(total), parent_type(0,0)
		// Pre: the operation is requested by all processes in the communicator comm
		{
			stopwatch::Time time(stopwatch::watches, "multivector::multivector");
			
//			world.barrier();
			
			//cout << "ID: " << world.rank() << ": new multivector" << endl;
			
			unsigned int wsize = world.size();
			
			// Compute everybody's portion
			{
				unsigned int share = total; // Number of vectors still to be distributed
				unsigned int pleft = wsize; // Number of processes still unemployed
				while (pleft)
				{
					p2gjb.push_back(share);
					share -= (share / pleft);
					p2gja.push_back(share);
					pleft--;
				}
				assert(share == 0);
				assert(pleft == 0);
			}
			
			assert(p2gja.size() == wsize);
			assert(p2gjb.size() == wsize);
			
			parent_type& mymat = *this;
			mymat.resize(size1, getsz());
			mymat.clear();
		}
	public:
		multivector& operator=(const multivector&) = default;
	public:
		matrix_column
		colio_local(unsigned int j)
		// Provides access to global column #j
		// Pre: the column is owned by the local process
		{
			assert(ismyj(j));
			unsigned int loclj = (j - getja());
			parent_type& x = *this;
			ublas::matrix_column< parent_type > mc(x, loclj);
			return mc;
		}
		
		void
		setcolumn(unsigned int j, const ublas::vector<value_type>& colmn)
		// Replaces global column #j
		// Pre: the column is owned by the local process
		{
			matrix_column mc = this->colio_local(j);
			mc = colmn;
		}
        
        ublas::vector<value_type>
        getcolumn(unsigned int j) const
		// Gets a copy of global column #j
		// Pre: the column is owned by the local process
        {
			assert(ismyj(j));
			unsigned int loclj = (j - getja());
			const parent_type& x = *this;
            ublas::vector<value_type> v = column(x, loclj);
			return v;
        }
		
		dense_matrix
		getwhole() const
        // Combines all portions to one dense matrix available locally
		// Pre: requested by all processes in the communicator of u
		{
			getcm().barrier();
			
			std::vector<parent_type> parts;
			mpi::all_gather(getcm(), (parent_type&)(*this), parts);
			dense_matrix whole(size1(), getot());
			for (auto p = 0; p < parts.size(); ++p)
			{
				ublas::range vrang(0, parts[p].size1());
				ublas::range hrang(p2gja[p], p2gjb[p]);
				ublas::matrix_range<dense_matrix> subma(whole, vrang, hrang);
				subma = parts[p];
			}
			return whole;
		}
	public:
		bool
		shrink_communicator()
		// If the local has no capacity, it is removed from the communicator
		// Returns true iff the local has any capacity
		{
			multivector& me = *this;
			
			unsigned int isfat = ((me.size2() == 0) ? 0 : 1);
			mpi::communicator small = me.getcm().split(isfat);
			
			embrace_communicator(small);
			
			return (isfat != 0);
		}
		
		void
		embrace_communicator(const mpi::communicator& newcm)
		{
			multivector& me = *this;
			
			unsigned int newja = me.getja();
			unsigned int newjb = me.getjb();
			
			assert(me.size2() == (newjb - newja));
			
			me.world = newcm;
			
			me.p2gja.resize(0);
			me.p2gjb.resize(0);
			mpi::all_gather(me.world, newja, me.p2gja);
			mpi::all_gather(me.world, newjb, me.p2gjb);
		}
		
	private:
		void
		keepj(unsigned int ja, unsigned int jb)
		{
            stopwatch::Time time(stopwatch::watches, "mypara::multivector::keepj()");

			multivector& me = *this;
            assert((0 <= ja) && (ja <= jb) && (jb <= me.getot()));

            unsigned int np = me.getcm().size();
            assert((np == me.p2gja.size()) && (np == me.p2gjb.size()));

            for (unsigned int p = 0; p != np; ++p)
            {
                // Old range of process p
                unsigned int oldja = me.p2gja[p];
                unsigned int oldjb = me.p2gjb[p];

                // New range of process p
                unsigned int newja, newjb;
                {
                    newja = std::min(std::max(oldja, ja), oldjb);
                    newjb = std::max(std::min(oldjb, jb), oldja);
                    assert(newja <= newjb);
                    assert((oldja <= newja) && (newjb <= oldjb));
                }

                // If I am process p, then cut my data to the new range
                if (p == me.getcm().rank()) {
                    ublas::range hrang(newja - oldja, newjb - oldja);
                    ublas::range vrang(0, me.size1());
                    ((parent_type)me) = ublas::matrix_range<parent_type>(me, vrang, hrang);
                }

                // Compute p2gja and p2gjb of process p
                {
                    assert(newja <= newjb);
                    if (newja != newjb) {
                        assert((ja <= newja) && (newja <= newjb) && (newjb <= jb));
                        newja -= ja;
                        newjb -= ja;
                    } else {
                        newja = newjb = 0;
                    }

                    me.p2gja[p] = newja;
                    me.p2gjb[p] = newjb;
                }
            }

            me.total = (jb - ja);
		}
		
	public:
		void
		split(multivector& A, multivector& B, unsigned int j0) const
		// First j0 columns of *this go to a, the others to b
		// Pre: *this, a and b all have the same communicator
		{
            stopwatch::Time time(stopwatch::watches, "mypara::multivector::split()");
			
			const multivector& me = *this;
			assert(j0 <= me.getot());
			
			//
			B.p2gja = A.p2gja = me.p2gja;
			B.p2gjb = A.p2gjb = me.p2gjb;
            B.total = A.total = me.total;
			
            // Clean data of a and b, but make consistent
			A.resize(0, A.getjb() - A.getja());
			B.resize(0, B.getjb() - B.getja());
			
            //
			A.keepj(0, j0);
			B.keepj(j0, me.getot());
			
			// Copy data
			A.resize(me.size1(), A.getjb() - A.getja());
			B.resize(me.size1(), B.getjb() - B.getja());
			
			// This could be done more efficiently:
			
			for (unsigned int j = A.getja(); j != A.getjb(); ++j) {
				A.setcolumn(j, me.getcolumn(j));
			}
			
			for (unsigned int j = B.getja(); j != B.getjb(); ++j) {
				B.setcolumn(j, me.getcolumn(j0 + j));
			}
		}
		
		void
		merge(multivector& a, multivector& b)
		// Pre: a and b are the (possibly modified) result of split
		// Note: a, b, and this need not share the communicator
		{
			stopwatch::Time time(stopwatch::watches, "mypara::multivector::merge()");
			
			multivector& me = *this;
			assert(me.getot() == (a.getot() + b.getot()));
			assert((me.size1() == a.size1()) && (me.size1() == b.size1()));
			
			for (unsigned int j = a.getja(); j != a.getjb(); ++j)
			{
				me.setcolumn(j, a.getcolumn(j));
			}
			
			unsigned int j0 = a.getot();
			for (unsigned int j = b.getja(); j != b.getjb(); ++j)
			{
				me.setcolumn(j0 + j, b.getcolumn(j));
			}
		}
		
	};
	
	// Operators
	
	double
	inner_prod_max(const multivector& a, const multivector& b)
	// Pre: a and b share the mpi communicator
	// Pre: the operation is requested by all processes in the communicator of a / b
	{
		stopwatch::Time time(stopwatch::watches, "multivector::inner_prod");
		
		std::vector<double> local_vec;
		{
			assert(a.getja() == b.getja());
			assert(a.getjb() == b.getjb());
			
			for (unsigned int k = 0; k != a.getsz(); ++k)
				local_vec.push_back(ublas::inner_prod(ublas::column(a, k), ublas::column(b, k)));
		}
		
		// negative infinity
		double local = std::numeric_limits<double>::lowest();
		
		if (local_vec.size())
			local = *std::max_element(local_vec.begin(), local_vec.end());
		
		double global = 0;
		{
			mpi::all_reduce(a.getcm(), local, global, mpi::maximum<double>());
		}
		
		return global;
	}
	
	double
	inner_prod(const multivector& a, const multivector& b)
	// Pre: a and b share the mpi communicator
	// Pre: the operation is requested by all processes in the communicator of a / b
	{
        stopwatch::Time time(stopwatch::watches, "multivector::inner_prod");
        
		double local = 0;
		{
			assert(a.getja() == b.getja());
			assert(a.getjb() == b.getjb());
            
			for (unsigned int k = 0; k != a.getsz(); ++k)
			{
				local += ublas::inner_prod(ublas::column(a, k), ublas::column(b, k));
			}
		}
        
		double global = 0;
		{
			mpi::all_reduce(a.getcm(), local, global, std::plus<double>());
		}
        
		return global;
	}
	
    // RIGHT MULTIPLY

    void
    prod_ref(const multivector& u, const multivector::dense_matrix& m, multivector& v)
    // This is the reference implementation for the following:
    // Right matrix multiply u * m, assumes no structure in m
    // Compare with function prod(u, m, jplan, v)
    // Pre: the operation is requested by all processes in the communicator of u and v
    // Pre: u and v have the same communicator
    {
        stopwatch::Time time(stopwatch::watches, "multivector::prod(ref)");

        typedef multivector::value_type value_type;
        typedef multivector::const_matrix_column const_matrix_column;
        typedef ublas::vector<value_type> vector;

        for (unsigned int j = 0; j != v.getot(); ++j)
        {
            namespace ublas = boost::numeric::ublas;
            typedef ublas::vector<value_type> Vec;

            // From the j-th column of the right matrix m, get local subvector
            ublas::vector<value_type> submv(u.getsz());
            {
                ublas::range verrg(u.getja(), u.getjb());
                const_matrix_column colmj(m, j);
                submv = ublas::vector_range<const_matrix_column>(colmj, verrg);
            }

            // Am I receiving data and/or sending data?
            bool itodo = (v.ismyj(j) || (0 != ublas::norm_inf(submv)));

            // A. Who has something to contribute?
            // Note: cannot interchange order with B
            mpi::communicator activ = u.getcm().split(itodo ? 1 : 0);

            // B. Do I have something to contribute? If not, skip.
            // Note: cannot interchange order with A
            if (!itodo) continue;

            // Find the rank of the process in the activ communicator
            // that holds the j-th column of the result
            unsigned int trank;
            {
                unsigned int local = (v.ismyj(j) ? activ.rank() : 0);
                mpi::all_reduce(activ, local, trank, std::plus<unsigned int>());
            }

            // Local contribution to j-th column of the result
            Vec myvec(u.size1());
            myvec = ublas::prod(u, submv);

            // Am I to receive the product vector?
            if (activ.rank() == trank) {
                Vec resul(myvec.size());
                mpi::reduce(activ, myvec, resul, std::plus<Vec>(), trank);
                v.setcolumn(j, resul);
            } else {
                mpi::reduce(activ, myvec, std::plus<Vec>(), trank);
            }
        }

        v.getcm().barrier();
    }

	
	struct Jplan
	{
	public:
		enum Strategy { nnz = 1, jforward = 2, jreverse = 4 };
	private:
		Strategy strategy;
	public:
		typedef double value_type;
		typedef ublas::compressed_vector<value_type> sparse_vector;
		typedef ublas::compressed_matrix<value_type, ublas::column_major> sparse_matrix;
	public:
		std::vector<sparse_vector> M_local;
	public:
		typedef ublas::vector<unsigned int> GROUP;
		typedef ublas::vector<unsigned int> TARGT;
	public:
		std::vector<GROUP> group_plan;
		std::vector<TARGT> targt_plan;
	private:
		bool ready;
	public:
		bool is_ready() const { return ready; }
	public:
		Jplan() : ready(false), strategy(Strategy::nnz) { }
		
		Jplan& operator()(const Strategy& s)
		{
			strategy = s;
			return (*this);
		}
		
		void
		compute(const multivector& u, const sparse_matrix& m, const multivector& v)
		// Pre: u and v have the same communicator
		{
			stopwatch::Time time(stopwatch::watches, "Jplan::compute");
			
			assert(u.getot() == m.size1());
			assert(v.getot() == m.size2());
			
			// Horizontal size of the result multivector v
			unsigned int width = (unsigned int)(m.size2());
			
			// Clear
			{
				M_local.clear();
				group_plan.clear();
				targt_plan.clear();
			}
			
			std::vector<sparse_vector> M;
			
			// Step 1.
			// Convert the right multiplication matrix to a vector of vectors
			for (unsigned int j = 0; j != width; ++j)
			{
				sparse_vector v(ublas::column(m, j));
				M.push_back(v);
				
				// From the j-th column of the right matrix m, get local subvector
				sparse_vector submv(u.getsz());
				{
					ublas::range range(u.getja(), u.getjb());
					submv = ublas::vector_range<sparse_vector>(v, range);
				}
				assert(submv.size() == u.getsz());
				M_local.push_back(submv);
			}
			
			// Step 2.
			// Create sets of disjoint communication groups
			{
				std::list<unsigned int> jleft;
				{
					std::vector<unsigned int> order(M.size());
					
					// Default strategy: Strategy::jforward
					std::iota(order.begin(), order.end(), 0.);
					
					if (strategy == Strategy::nnz)
					{
						// Sort with decreasing nnz value
						struct R {
							typedef std::vector<sparse_vector> VOV;
							const VOV& M;
							R(const VOV& M) : M(M) { }
							bool operator()(unsigned int i, unsigned int j) const {
								return (M[i].nnz() > M[j].nnz());
							}
						};
						std::sort(order.begin(), order.end(), R(M));
					}
					
					if (strategy == Strategy::jreverse) {
						std::sort(order.rbegin(), order.rend());
					}
					
					for (unsigned int j = 0; j != width; ++j)
						jleft.push_back(order[j]);
				}
				
				// While there are target columns to process
				while (jleft.size())
				{
					GROUP group(u.getcm().size());
					// group[p] is the number of the communication group
					// where process p is involved
					// It is zero if the process has nothing to communicate
					
					TARGT targt(u.getcm().size());
					// targt[p] is the target column in the result
					// for the communication group masks[p]
					
					group.clear();
					targt.clear();
					
					unsigned int grpno = 0;
					auto pj = jleft.begin();
					while (pj != jleft.end())
					{
						// jmask[p] is one if process p is involved, and zero else
						ublas::vector<unsigned int> jmask(group.size());
						jmask.clear();
						
						unsigned int j = *pj;
						
						// conflict iff jmask has nontrivial overlap with group
						bool conflict = false;
						
						// Target index
						{
							unsigned int p = v.gj2pr(j);
							conflict = conflict || (group[p] != 0);
							jmask[p] = 1;
						}
						
						// Source indices
						{
							typedef ublas::compressed_vector<double> sparse_vector;
							const sparse_vector& vj = M[j];
							// (Does a non-const iterator work here?)
							for (sparse_vector::const_iterator i = vj.begin(); i != vj.end(); ++i)
							{
								unsigned int p = u.gj2pr((unsigned int)i.index());
								conflict = conflict || (group[p] != 0);
								jmask[p] = 1;
								if (conflict) break;
							}
						}
						
						if (conflict) {
							// Bad luck, try another column j
							pj++;
						} else {
							grpno++;
							group += (grpno * jmask);
							targt += (j * jmask);
							
							// The target column j has been scheduled for processing
							// Remove from queue and proceed to next column
							pj = jleft.erase(pj);
						}
					}
					
					group_plan.push_back(group);
					targt_plan.push_back(targt);
				}
			}
			
			this->ready = true;
		}
	}; // class Jplan
	
	
	void
	prod(const multivector& u, const Jplan::sparse_matrix& m, Jplan& jplan, multivector& v)
    // Right matrix multiply, u * m
	// Good for matrices m where each row is sparse
	// Pre: the operation is requested by all processes in the communicator of u and v
	// Pre: u and v have the same communicator
	// TODO: implement using MPI groups
	{
        stopwatch::Time time(stopwatch::watches, "multivector::prod(jplan)");
		
		typedef multivector::value_type value_type;
		//		typedef multivector::dense_matrix dense_matrix;
		typedef multivector::const_matrix_column const_matrix_column;
        typedef ublas::vector<value_type> vector;
		
		if (!jplan.is_ready()) jplan.compute(u, m, v);
		assert(jplan.is_ready());
		
		for (unsigned int k = 0; k != jplan.group_plan.size(); ++k)
		{
			auto& group = jplan.group_plan[k];
			auto& targt = jplan.targt_plan[k];
			
			//std::cout << "group: " << group << std::endl;
			//std::cout << "targt: " << targt << std::endl;
			
			// Partition the MPI communicator into disjoint groups
			
			unsigned int r = u.getcm().rank();
			unsigned int color = group[r];
			//std::cout << masks << std::endl;
			mpi::communicator activ = u.getcm().split(color);
			
			//std::cout << "Proc: " << r << "; color: " << color << "; targt: " << targt[r] << std::endl;
			
			// Am I in one of the communicating groups?
			if (!color) continue;
			
			// The truly parallel part:
			// Compute the partial matrix matrix product
			
			unsigned int j = targt[r];
			
			// From the j-th column of the right matrix m, get local subvector
			const Jplan::sparse_vector& submv = jplan.M_local[j];
			
			// Find the rank of the process in the activ communicator
			// that holds the j-th column of the result
			unsigned int jdest;
			{
				unsigned int local = (v.ismyj(j) ? activ.rank() : 0);
				mpi::all_reduce(activ, local, jdest, std::plus<unsigned int>());
			}
			
			//std::cout << "j: " << j << "; trank: " << trank << std::endl;
			
			// Local contribution to j-th column of the result
			typedef ublas::vector<value_type> Vec;
			Vec myvec(u.size1()); myvec.clear();
			myvec = ublas::prod(u, submv);
			
			// Am I to receive the product vector?
			if (activ.rank() == jdest) {
				Vec resul(myvec.size());
				mpi::reduce(activ, myvec, resul, std::plus<Vec>(), jdest);
				v.setcolumn(j, resul);
			} else {
				mpi::reduce(activ, myvec, std::plus<Vec>(), jdest);
			}
		}

        #ifndef NDEBUG
        /// Check correctness
        {
            multivector v_ref = v;
            mypara::prod_ref(u, m, v_ref);
            assert(v.size1() == v_ref.size1());
            assert(v.size2() == v_ref.size2());
            assert(ublas::norm_inf(v - v_ref) <= 1e-6);
        }//*/
        #endif
	}
	
	multivector
	prod(const multivector& u, const multivector::dense_matrix& m, Jplan& jplan)
    // Right matrix multiply. See void prod(u, m, jplan, v)
	{
		assert(u.getot() == m.size1());
		
		multivector v(u.size1(), (unsigned int)(m.size2()), u.getcm());
		mypara::prod(u, m, jplan, v);
		
		return v;
	}
	

	void
	prod(const multivector& u, const ublas::compressed_matrix<double, ublas::column_major>& m, multivector& v)
	// Asynchronous implementation of:
    // Right matrix multiply u * m
	// Compare with function prod(u, m, jplan, v)
    // Pre: m is sparse (send buffers created)
	// Pre: the operation is requested by all processes in the communicator of u and v
	// Pre: u and v have the same communicator
	{
        stopwatch::Time time(stopwatch::watches, "multivector::prod(asynch)");

        v.clear();
		
		// Type of m
		typedef ublas::compressed_matrix<double, ublas::column_major> matrix;
		// Column of m
		typedef ublas::matrix_column<const matrix> cmc;
		// Vector type of column of m
		typedef ublas::compressed_vector<matrix::value_type> sparse_vector;
			
		// Step 1.
		// Convert the right multiplication matrix to a vector of vectors
		std::vector<sparse_vector> M, M_local;
		{
			stopwatch::Time time(stopwatch::watches, "multivector::prod(asynch).Mlocal");
			
			for (unsigned int j = 0; j != m.size2(); ++j)
			{
				sparse_vector vcolj(ublas::column(m, j));
				
				// From the j-th column of the right matrix m, get local subvector
				sparse_vector submv(u.getsz());
				{
					ublas::range range(u.getja(), u.getjb());
					submv = ublas::vector_range<sparse_vector>(vcolj, range);
					assert(submv.size() == u.getsz());
				}
				
				M.push_back(vcolj);
				M_local.push_back(submv);
			}
		}
		
		// Vector for a column of u or v
		typedef ublas::vector<multivector::value_type> Vec;

        //
        struct Packet {
            int dest, src, tag;
            Vec myvec;
            //
            Packet(int dest, int tag, const Vec& myvec, int src) : dest(dest), tag(tag), myvec(myvec), src(src)
            { }
			//
			std::string str() const {
                std::stringstream s;
                s << "dest: " << dest << ", "
                  << "tag: " << tag << ", "
                  << "vecsize: " << myvec.size() << ", "
                  << "src: " << src;
                return s.str();
			}
        };

        // Queue in everything that has to be sent
        std::list<Packet> sendbuf;
        std::vector<mpi::request> sendreq;
		{
			stopwatch::Time time(stopwatch::watches, "multivector::prod(asynch).sendbuf");
			
			// Prepare for send: iterate over target columns
			for (unsigned int j = 0; j != v.getot(); ++j)
			{
				// Is the target local?
				if (v.ismyj(j)) continue;
				
				// Local subvector from the j-th column of the right matrix m
				const sparse_vector& submv = M_local[j];
				
				// Do I have data to send?
				if (0 == submv.nnz()) continue;
				
				// Local contribution to j-th column of the result
				Vec myvec = ublas::prod(u, submv);
				assert(myvec.size() == v.size1());
				
				// Destination rank
				const unsigned int trank = v.gj2pr(j);
				// Source rank
				const unsigned int srank = u.getcm().rank();
				
				assert(trank != srank);
				const int tag = j;
				// Put data into send buffer
				sendbuf.push_back(Packet(trank, tag, myvec, srank));
				
				// Submit MPI send request
				const Packet& packet = sendbuf.back();
				//std::cout << "Sending packet " << packet.str() << std::endl;
				sendreq.push_back(u.getcm().isend(packet.dest, packet.tag, packet.myvec));
			}
		}
		
        // Queue in everything that has to be received
        std::list<Packet> recvbuf;
        std::vector<mpi::request> recvreq;
		{
			stopwatch::Time time(stopwatch::watches, "multivector::prod(asynch).recvbuf");
			
			for (unsigned int j = v.getja(); j != v.getjb(); ++j)
			{
				// j-th column of m
				sparse_vector vcolj = M[j];
				
				// Identify senders
				std::set<unsigned int> ranks;
				for (sparse_vector::const_iterator i = vcolj.begin(); i != vcolj.end(); ++i)
				{
					// Is the source local?
					if (u.ismyj((unsigned int)i.index())) continue;
					
					unsigned int srank = u.gj2pr((unsigned int)i.index());
					ranks.insert(srank);
				}
				
				for (auto i = ranks.begin(); i != ranks.end(); ++i)
				{
					const unsigned int srank = (*i);
					const int tag = j;
					
					// Receive buffer
                    // The target vector will be automatically resized by boost::mpi
					recvbuf.push_back(Packet(v.getcm().rank(), tag, Vec(0), srank));
					
					// Submit MPI receive request
					Packet& packet = recvbuf.back();
					//std::cout << "Recving packet " << packet.str() << std::endl;
					recvreq.push_back(u.getcm().irecv(packet.src, packet.tag, packet.myvec));
				}
			}
		}
		
        // Do what is entirely local
		{
			stopwatch::Time time(stopwatch::watches, "multivector::prod(asynch).local");
			
			for (unsigned int j = v.getja(); j != v.getjb(); ++j)
			{
				// Local contribution to j-th column of the result
				v.setcolumn(j, ublas::prod(u, M_local[j]));
			}
		}
		
		// Receive all data
		{
			stopwatch::Time time(stopwatch::watches, "multivector::prod(asynch).recvall");
			
			mpi::wait_all(recvreq.begin(), recvreq.end());
			//std::cout << "#" << v.getcm().rank() << ": recieved" << std::endl;
			
			for (std::list<Packet>::const_iterator i = recvbuf.begin(); i != recvbuf.end(); ++i)
			{
				unsigned int j = i->tag;
				assert(v.ismyj(j));
				assert(i->myvec.size() == v.size1());
				v.colio_local(j) += i->myvec;
			}
		}
		
		// Wait for all sent data to be received (before destroying the send buffer)
		{
			stopwatch::Time time(stopwatch::watches, "multivector::prod(asynch).sendall");
			
			mpi::wait_all(sendreq.begin(), sendreq.end());
			//std::cout << "#" << v.getcm().rank() << ": sent" << std::endl;
		}
//		v.getcm().barrier();

        #ifndef NDEBUG
        /// Check correctness
        {
            multivector v_ref = v;
            mypara::prod_ref(u, m, v_ref);
            assert(v.size1() == v_ref.size1());
            assert(v.size2() == v_ref.size2());
            assert(ublas::norm_inf(v - v_ref) <= 1e-6);
			std::cout << "Prod check ok (use -DNDEBUG to disable)" << std::endl;
        }//*/
        #endif
	}
	
	multivector
    prod(const multivector& u, const ublas::compressed_matrix<double, ublas::column_major>& m)
    // Right matrix multiply. See void prod(u, m, v)
	{
		assert(u.getot() == m.size1());
		
		multivector v(u.size1(), (unsigned int)(m.size2()), u.getcm());
		mypara::prod(u, m, v);
		
		return v;
	}


    // LEFT MULTIPLY
	
	
	multivector
	prod(const ublas::compressed_matrix<double, ublas::row_major>& m, const multivector& u)
    // Left matrix multiply
	{
        stopwatch::Time time(stopwatch::watches, "multivector::prod(left)");
		
//		u.getcm().barrier();
		
		assert(m.size2() == u.size1());
		
		
		multivector v(m.size1(), u.getot(), u.getcm());
		{
			//ublas::compressed_matrix<multivector::value_type, ublas::column_major> m1(m);
			//ublas::matrix<multivector::value_type, ublas::row_major> u1(u);
			stopwatch::Time time(stopwatch::watches, "multivector::prod(left)/ublas::prod");
			((multivector::parent_type&)v) = boost::numeric::ublas::prod(m, u);
		}
		
		return v;
	}
}



#endif
