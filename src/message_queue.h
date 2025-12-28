#pragma once
#include <queue>
#include <mutex>
#include <condition_variable>

template <typename T>
class message_queue {
private:
	mutable std::mutex m;
	std::queue<T> q;
	std::condition_variable cond_var;
	using queue_lock = std::unique_lock<std::mutex>;

public:
	void push(T new_value) {
		queue_lock lock(m);
		q.push(std::move(new_value));
		cond_var.notify_one();
	}

	T pop() {
		queue_lock lock(m);
		cond_var.wait(lock, [this] { return !q.empty(); });
		T res = std::move(q.front());
		q.pop();
		return res;
	}

	bool empty() const {
		queue_lock lock(m);
		return q.empty();
	}
};
