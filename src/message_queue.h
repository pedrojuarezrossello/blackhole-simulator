#ifndef MESSAGE_QUEUE_H
#define MESSAGE_QUEUE_H

#include <new>
#include <atomic>
#include <limits>
#include <thread>
#include <cstddef>

/*
* We'll allocate a chunk of memory for the queue and use it as a circular buffer,
* this way we won't have to worry about allocations and deallocations which are very expensive as per profiler.
* We'll use two atomic indices to keep track of the write and read positions in the buffer and
* a state array to keep track of whether the slot is empty or full.
* Also this is a single producer single consumer queue, so we're not using compare_exchange_strong etc.
* Note that we want to reuse the memory so we assume that the type T has a set() method that will
* change the value of the object without allocating new memory, this way we can reuse the same slot in the buffer.
*/

constexpr size_t MAX_QUEUE_SIZE = std::numeric_limits<unsigned char>::max() + 1; // 256

template <typename T, bool settable = true>
class fixed_sized_message_queue {
	enum state : bool {
		empty = 0,
		full = 1
	};
	// Avoid false sharing by aligning the indices to the size of a cache line
	alignas(std::hardware_destructive_interference_size) T * buffer;
	// TODO Since both threads will look up the state in close proximity, they'll be in the same cache block so we have to somehow space them out...
	alignas(std::hardware_destructive_interference_size) std::atomic<state> states_[MAX_QUEUE_SIZE] = {};
	alignas(std::hardware_destructive_interference_size) std::atomic<unsigned char> write_;
	alignas(std::hardware_destructive_interference_size) std::atomic<unsigned char> read_;

public:

	fixed_sized_message_queue()
		: buffer(new T[MAX_QUEUE_SIZE] {})
		, write_(0)
		, read_(0) {
	
		}

	template <typename... Args>
	void push(Args&&... args) {
		auto curr_write = write_.load(std::memory_order_relaxed);
		write_.store(curr_write+1, std::memory_order_relaxed);

		while (states_[curr_write].load(std::memory_order_acquire) == state::full)
			std::this_thread::yield();

		if constexpr (settable)
			buffer[curr_write].set(std::forward<Args>(args)...);
		else
			buffer[curr_write] = T(std::forward<Args>(args)...);
		
		states_[curr_write].store(state::full, std::memory_order_release);
	}

	T pop() {
		auto curr_read = read_.load(std::memory_order_relaxed);
		read_.store(curr_read + 1, std::memory_order_relaxed);

		while (states_[curr_read].load(std::memory_order_acquire) != state::full)
			std::this_thread::yield();
		
		T element = std::move(buffer[curr_read]);
		states_[curr_read].store(state::empty, std::memory_order_release);
		return element;
	}

	bool is_empty() const {
		return write_.load(std::memory_order_acquire) == read_.load(std::memory_order_acquire);
	}
};

#endif
