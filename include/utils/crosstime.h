
/*
	Easy embeddable cross-platform high resolution timer function. For each 
	platform we select the high resolution timer. You can call the 'time_ns()' 
	function in your file after embedding this.
	https://gist.github.com/nickolayl/f6651e978db145141faa1d986182014f
*/

#include <stdint.h>
#if defined(__APPLE__)
	#include <mach/mach_time.h>
#elif defined(_WIN32)
	#include <windows.h>
#else
	#ifdef _POSIX_C_SOURCE
		#define NS_OLD_POSIX_C_SOURCE _POSIX_C_SOURCE
		#undef _POSIX_C_SOURCE
	#endif
	#define _POSIX_C_SOURCE 199309L
	#ifndef __USE_POSIX199309
		#define __USE_POSIX199309
		#define NS_DEFINED__USE_POSIX199309
	#endif
	#include <time.h>
	#ifdef NS_DEFINED__USE_POSIX199309
		#undef __USE_POSIX199309
	#endif
	#ifdef NS_OLD_POSIX_C_SOURCE
		#undef _POSIX_C_SOURCE
		#define _POSIX_C_SOURCE NS_OLD_POSIX_C_SOURCE
	#endif
	#ifdef CLOCK_MONOTONIC
		#define NS_CLOCK_ID CLOCK_MONOTONIC
	#else
		#define NS_CLOCK_ID CLOCK_REALTIME
	#endif
#endif
static inline uint64_t time_ns(void) {
	static uint8_t ns_inited = 0;
	#if defined(__APPLE__)
		static mach_timebase_info_data_t info;
		uint64_t now;
		if (ns_inited == 0) {
			mach_timebase_info(&info);
			ns_inited = 1;
		}
		now = mach_absolute_time();
		now *= info.numer;
		now /= info.denom;
		return now;
	#elif defined(_WIN32)
		static LARGE_INTEGER win_frequency;
		LARGE_INTEGER now;
		if (ns_inited == 0) {
			QueryPerformanceFrequency(&win_frequency);
			ns_inited = 1;
		}
		QueryPerformanceCounter(&now);
		return (uint64_t)((1.0e9 * now.QuadPart) / win_frequency.QuadPart);
	#else
		static struct timespec linux_rate;
		uint64_t now;
		struct timespec spec;
		if (ns_inited == 0) {
			clock_getres(NS_CLOCK_ID, &linux_rate);
			ns_inited = 1;
		}
		clock_gettime(NS_CLOCK_ID, &spec);
		now = spec.tv_sec * (uint64_t)1.0e9 + spec.tv_nsec;
		return now;
	#endif
}
