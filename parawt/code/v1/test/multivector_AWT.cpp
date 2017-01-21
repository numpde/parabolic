//
//  multivector_AWT.cpp
//
//  Created by r. on 18/05/14
//

// Disables checks in boost
#ifndef NDEBUG
#define NDEBUG
#endif

//
#include "../include/reporter.hpp"

#include <iostream>
#include "../include/stopwatch.hpp"
#include "../include/multivector.hpp"
#include "../include/multivector_AWT.hpp"
#include "../include/tFEM.hpp"
#include "../include/AWT.hpp"

#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>

namespace mypara
{
	using namespace std;
	namespace mpi = boost::mpi;
	namespace ublas = boost::numeric::ublas;
	
	void test_multivector_AWT_basic()
	{
		cout << "Enter: mypara::test_multivector_AWT_basic()" << endl;
		stopwatch::Time time(stopwatch::watches, "mypara::test_multivector_AWT_basic()");
		
		mpi::communicator world;
		
		typedef ublas::matrix<double> matrix;
		
		using namespace mypara;
		
		unsigned int size1 = 3; // Number of rows
		unsigned int size2 = 9; // Number of columns
		
		
		// Create AWT
		double T = 4;
		spacetime::tmesh te;
		te.makeuniform(0, T, size2);
		
		spacetime::tFEM tfem(te);
		
		unsigned int nu = 2;
		mypara::AWT awt(tfem.AtE, tfem.MtE, nu);
		
		// Check generalized "eigenvalues" of awt
		{
			spacetime::AWT awt2(tfem.AtE, tfem.MtE, nu);
			bool ok = true;
			for (auto i = 0; i != awt.gamma.size(); ++i) {
				ok = ok && (fabs(awt.gamma[i] - awt2.gamma[i]) <= 1e-8);
			}
			cout << "Eigenvalues OK? " << (ok ? "YES" : "NO") << endl;
			assert(ok);
		}
		
		matrix v_ref(size1, size2); v_ref.clear();
		// Create matrix to be distributed
		{
			for (unsigned int j = 0; j != size2; ++j)
			{
				v_ref(j % size1, j) = 1;
//				v_ref(j, j) = 1;
			}
		}
		
		multivector v(size1, size2, world);
		// Distributed matrix
		{
			for (unsigned int j = v.getja(); j < v.getjb(); ++j) {
				v.colio_local(j) = ublas::column(v_ref, j);
			}
		}
		
		// Check multivector::get_whole()
		{
			matrix z = (v_ref - v.getwhole());
			bool ok = (ublas::norm_frobenius(z) == 0);
			cout << "Get whole OK? " << (ok ? "YES" : "NO") << endl;
			assert(ok);
			world.barrier();
		}
		
		// Print info
		{
			for (unsigned int j = v.getja(); j < v.getjb(); ++j) {
				cout << "Proc " << world.rank() << ": " << "column #" << j << " is " << v.colio_local(j) << endl;
			}
			cout.flush(); world.barrier();
		}
		
		// Multiplication 1
		{
			// Print info
			{
				if (!world.rank()) cout << "Testing right multiplication by T" << endl;
				cout.flush(); world.barrier();
			}
			
			multivector w = awt.uT(v);
			
			matrix w_ref;
			{
				spacetime::AWT awt(tfem.AtE, tfem.MtE, nu);
				w_ref = ublas::prod(v_ref, awt.V);
			}
			
			// Print info
			{
				for (unsigned int j = w.getja(); j < w.getjb(); ++j) {
					cout << "Proc " << world.rank() << ": " << "column #" << j << " is " << w.colio_local(j) << endl;
				}
				
				multivector::dense_matrix w_whole = w.getwhole();
				if (!world.rank()) cout << "w_whole: " << w_whole << endl;
				
				cout.flush(); world.barrier();
			}
			
			// Check multiplication result
			{
				if (!world.rank()) cout << "w_ref: " << w_ref << endl;
				matrix z = (w_ref - w.getwhole());
				bool ok = (ublas::norm_frobenius(z) <= 1e-10);
				cout << "Multiplication 1 OK? " << (ok ? "YES" : "NO") << endl;
				assert(ok);
			}
		}
		
		// Multiplication 2
		{
			// Print info
			{
				if (!world.rank()) cout << "Testing right multiplication by Tt" << endl;
				cout.flush(); world.barrier();
			}
			
			multivector w = awt.uTt(v);
			
			matrix w_ref;
			{
				spacetime::AWT awt(tfem.AtE, tfem.MtE, nu);
				w_ref = ublas::prod(v_ref, ublas::trans(awt.V));
			}
			
			// Print info
			{
				for (unsigned int j = w.getja(); j < w.getjb(); ++j) {
					cout << "Proc " << world.rank() << ": " << "column #" << j << " is " << w.colio_local(j) << endl;
				}
				
				multivector::dense_matrix w_whole = w.getwhole();
				if (!world.rank()) cout << "w_whole: " << w_whole << endl;
				
				cout.flush(); world.barrier();
			}
			
			// Check multiplication result
			{
				if (!world.rank()) cout << "w_ref: " << w_ref << endl;
				matrix z = (w_ref - w.getwhole());
				bool ok = (ublas::norm_frobenius(z) <= 1e-10);
				cout << "Multiplication 2 OK? " << (ok ? "YES" : "NO") << endl;
				assert(ok);
			}
		}
		
		//
		{
			multivector::dense_matrix v_whole = v.getwhole();
			
			// Print info
			{
				if (!world.rank()) cout << "v_whole: " << v_whole << endl;
				cout.flush(); world.barrier();
			}
			
		}
		
		cout << "Exit: mypara::test_multivector_AWT_basic()" << endl;
	}
	
//	void test_multivector_AWT_large()
//	{
//		cout << "Enter: mypara::test_multivector_AWT_large()" << endl;
//		stopwatch::Time time(stopwatch::watches, "mypara::test_multivector_AWT_large()");
//		
//		mpi::communicator world;
//		
//		unsigned int size1 = 100000;
//		unsigned int size2 = (1 << 12);
//		
//		multivector v(size1, size2, world);
//		
//		{
//			typedef ublas::compressed_matrix<double> sparse_matrix;
//			
//			sparse_matrix m(size2, size2, 3*size2); m.clear();
//			m = ublas::identity_matrix<>(size2);
//			
//			stopwatch::StopWatch w;
//			w.tic();
//			v * m;
//			w.add();
//			
//			// Print timings
//			{
//				cout.flush(); world.barrier();
//				if (!world.rank())
//				{
//					cout << "@iden: ";
//					cout << "size2: " << size2 << " ";
//					cout << "time(ms): " << w.ms().count() << " ";
//					cout << endl;
//				}
//				world.barrier();
//			}
//		}
//		
//		cout << "Exit: mypara::test_multivector_AWT_large()" << endl;
//	}
	
	void test_multivector_AWT_scale()
	{
		cout << "Enter: mypara::test_multivector_AWT_scale()" << endl;
		stopwatch::Time time(stopwatch::watches, "mypara::test_multivector_AWT_scale()");
		
		reporter::note("Parallel AWT scalability test");
		reporter::note("Timings are in milliseconds");
		
		typedef ublas::matrix<double> matrix;
		typedef ublas::compressed_matrix<double> sparse_matrix;
		
		mpi::communicator world;
		
		//unsigned int n1 = 12; // size1 = dimV increases up to (2 ^ n1)
		unsigned int n2 = 14; // size2 = dimE increases up to (2 ^ n2)
		
		//for (unsigned int size1 = (1 << n1); size1 <= (1 << n1); size1 *= 2)
		unsigned size1 = 1953;
		reporter::note["dimV"] = size1;
		reporter::note("Number of temporal elements")["N = []"];
		reporter::note("Multiplication by T [ms]")["T = []"];
		reporter::note("Multiplication by V [ms]")["V = []"];
		{
			for (unsigned int size2 = 1; size2 <= (1 << n2); size2 *= 2)
			{
				reporter::note[""];
				reporter::note["n"] = size2;
				
				multivector v(size1, 1 + size2, world);
				
				// Create AWT
				double T = 2;
				spacetime::tmesh te;
				te.makeuniform(0, T, v.getot());
				
				spacetime::tFEM tfem(te);
				
				unsigned int nu = 2;
				AWT awt(tfem.AtE, tfem.MtE, nu);
				
				double t_max = 0;
				
				// Multiply by T
				{
					stopwatch::StopWatch w;
					//awt.uT(v); // compute jplan
					w.tic();
					awt.uT(v);
					w.add();
					double t = w.ms().count();
					t_max = max(t, t_max);
					
					// Print timings
					{
						cout.flush(); world.barrier();
						if (!world.rank())
						{
							cout << "@T: ";
							cout << "size1: " << size1 << " ";
							cout << "size2: " << size2 << " ";
							cout << "time(ms): " << t << " ";
							cout << endl;
						}
						world.barrier();
					}
					
					reporter::note["t"] = t;
				}
				
				// Multiply by Tt
				{
					stopwatch::StopWatch w;
					//awt.uTt(v); // compute jplan
					w.tic();
					awt.uTt(v);
					w.add();
					double v = w.ms().count();
					t_max = max(v, t_max);
					
					// Print timings
					{
						cout.flush(); world.barrier();
						if (!world.rank())
						{
							cout << "@V: ";
							cout << "size1: " << size1 << " ";
							cout << "size2: " << size2 << " ";
							cout << "time(ms): " << v << " ";
							cout << endl;
						}
						world.barrier();
					}
					
					reporter::note["v"] = v;
				}
				
				
				reporter::note["N.append(n);"];
				reporter::note["T.append(t);"];
				reporter::note["V.append(v);"];
				
				// Abort loop if the trafo is taking too long
				double t_max_ok = 4 * 60e3; // seconds
				if (t_max >= t_max_ok) break;
			} // for size2
		} // for size1
		
		cout << "Exit: mypara::test_multivector_AWT_scale()" << endl;
	}
}


void test()
{
	reporter::note.is_quiet = (boost::mpi::communicator().rank() != 0);
	
	mypara::test_multivector_AWT_basic();
//    mypara::test_multivector_AWT_large();
    mypara::test_multivector_AWT_scale();
}
