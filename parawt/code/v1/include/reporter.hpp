//
//  reporter.hpp
//  round1
//
//  Created by r. on 18/07/15.
//  Copyright (c) 2015 LJLL. All rights reserved.
//

// Example usage:
//
// unsigned K = 123, N = 3;
// reporter::note("Temporal elements") << "K = " << K << std::endl;
// reporter::note("Spatial refinements") << "N = " << N << std::endl;
//
// produces:
//
// # Temporal elements
// K = 123
// # Spatial refinements
// N = 3

#ifndef round1_reporter_hpp
#define round1_reporter_hpp

#include <iostream>
#include <sstream>
#include <vector>

namespace reporter {
	struct reporter_class : public std::stringstream
	{
	public:
		struct catch_the_operator {
		public:
			reporter_class& r;
			const std::string var;
			bool line_break_due = true;
		private:
			void issue_line_break() {
				if (line_break_due) r << std::endl;
				line_break_due = false;
			}
		public:
			catch_the_operator(reporter_class& r, const std::string& var) : r(r), var(var) {
				r << var;
			}
			
			~catch_the_operator() {
				issue_line_break();
			}
		public:
			template <class T>
			reporter_class&
			operator=(const T& val) {
				r << " = " << val;
				return r;
			}
			
			template <class T>
			reporter_class&
			operator=(const std::vector<T>& val) {
				r << " = ";
				r << "[";
				for (auto i = val.begin(); i != val.end(); ++i)
					r << (i != val.begin() ? ", " : "") << (*i);
				r << "]";
				return r;
			}
			
			catch_the_operator
			operator[](const std::string& var) {
				issue_line_break();
				return catch_the_operator(r, var);
			}
		};
	public:
		~reporter_class() {
			report();
		}
	public:
		bool is_quiet = false;
	private:
		void report() const {
			if (is_quiet) return;
			
			std::cout << "# REPORTER BEGIN ------" << std::endl;
			std::cout << (*this).str() << std::endl;
			std::cout << "# REPORTER END --------" << std::endl;
		}
	public:
		reporter_class&
		operator()(const std::string& comment) {
			(*this) << (comment.size() ? "# " : "") << comment << std::endl;
			return (*this);
		}
	public:
		catch_the_operator
		operator[](const std::string& var) {
			return catch_the_operator(*this, var);
		}
	} note;
}

#endif
