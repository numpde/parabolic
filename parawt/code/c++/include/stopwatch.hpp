//
//  stopwatch.hpp
//
//  Created by r. on 08/05/14
//

#ifndef round1_stopwatch_hpp
#define round1_stopwatch_hpp

#include <iostream>
#include <map>
#include <chrono>
#include <thread>
#include <string>

namespace stopwatch
{
	using namespace std;
	
    class StopWatch
    {
	private:
    private:
        mutable chrono::high_resolution_clock::duration total;
        mutable chrono::high_resolution_clock::time_point a;
    private:
        // Should be hidden
		mutable long long counter; // counts calls to tic()
	private:
		mutable int rec;
    public:
        StopWatch() : total(0), a(chrono::high_resolution_clock::now()), counter(0), rec(0) { }
        
        void tic() const { counter++; if (rec == 0) a = chrono::high_resolution_clock::now(); }
        void add() const { if (rec == 0) total += (chrono::high_resolution_clock::now() - a); }
	public:
		void rec_more() const { rec++; }
		void rec_less() const { rec--; }
        
        chrono::milliseconds ms() const { return chrono::duration_cast<chrono::milliseconds>(total); }
	public:
		static void report(const std::map<std::string, StopWatch>& watches)
		{
			cout << "Stopwatch report: " << endl;
			
			if (watches.size() == 0) {
				cout << "No stopwatches. Nothing to report." << endl;
				return;
			}
			
			unsigned int n = 1;
			for (auto i = watches.begin(); i != watches.end(); ++i, ++n)
			{
				cout << n << ". ";
				auto w = (i->second);
				cout << (i->first) << ". ";
				cout << "Calls: " << w.counter << ". ";
				cout << "Total: " << w.ms().count() << " ms" << ". ";
				cout << "Averg: " << (((double)w.ms().count()) / w.counter) << " ms" << ".";
				cout << endl;
			}
		}
	public:
		static void pause_ms(unsigned int ms) {
			this_thread::sleep_for(chrono::milliseconds(ms));
		}
    };
	
	// !
	extern std::map<std::string, StopWatch> watches;
	
    struct Time
    {
	public:
		typedef std::map<std::string, StopWatch> Watches;
		typedef std::string Key;
	private:
		mutable chrono::high_resolution_clock::time_point born;
    private:
        const StopWatch& w;
    public:
        Time(Watches& watches, const Key& key)
        :
        w(watches.insert(make_pair(key, StopWatch())).first->second),
		born(chrono::high_resolution_clock::now())
        {
            w.tic();
			w.rec_more(); // take care of recursive functions constructing Time()
        }
        
        ~Time()
        {
			w.rec_less(); // take care of recursive functions
            w.add();
        }
	public:
		double age_ms() const {
			return
			(
				chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - born)
			)
			.count();
		}
    };
	
} // namespace stopwatch

#endif
