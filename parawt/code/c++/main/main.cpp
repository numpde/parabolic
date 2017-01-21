//
//  main.cpp
//
//  Created by r. on 08/05/14
//

// Compile with something like
//
//	g++ $STANDARD main.cpp ../test/multivector_AWT.cpp $INCLUDES $LIBPATHS $LINKLIBS $COMPFLAG
//
// where
//
//	$STANDARD = -std=c++11
//	$INCLUDES = -I/usr/lib64/mpi/gcc/openmpi/include/
//	$LIBPATHS = -L/usr/lib64/mpi/gcc/openmpi/lib64/
//	$LINKLIBS = -lmpi -lmpi_cxx -lboost_mpi-mt -lboost_serialization-mt
//	$COMPFLAG = -D_GLIBCXX_USE_NANOSLEEP -O1

#include <boost/mpi.hpp>
#include <iostream>

#include "../include/stopwatch.hpp"
namespace stopwatch
{
	std::map<std::string, StopWatch> watches;
}

// From some linked file
extern void test();

// Entry point
int main()
{
	#ifdef BOOST_MPI_HPP
	boost::mpi::environment mpienv;
	
	test();
	
	// Report
	{
		using namespace std;
		boost::mpi::communicator world;
		for (unsigned int r = 0; r != world.size(); ++r)
		{
			cout.flush(); world.barrier();
			if (r != world.rank()) continue;
			
			cout << "Proc #" << r << ":" << endl;
			stopwatch::StopWatch::report(stopwatch::watches);
		}
	}
	#else
	test();
	stopwatch::StopWatch::report(stopwatch::watches);
	#endif
	
    return EXIT_SUCCESS;
}
