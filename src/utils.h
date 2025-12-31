#pragma once
#include <vector>
#include <string_view>
#include <type_traits>
#include <fstream>


template <typename T, typename F, std::size_t... Is, typename ... Args>
constexpr std::array<std::decay_t<T>, sizeof...(Is)>
impl_create_array(std::index_sequence<Is...>, F && fn, Args &&... args) {
	return { { (static_cast<void>(Is), std::forward<F>(fn)(Is, std::forward<Args>(args)...))... } };
}

template <typename T, std::size_t N, typename F, typename... Args>
constexpr std::array<std::decay_t<T>, N> create_array(F&& fn, Args&&... args) {
	return impl_create_array<T>(std::make_index_sequence<N>(), std::forward<F>(fn), std::forward<Args>(args)...);
}

enum class spacetime {
	schwarzschild,
	kerr
};

struct shared_initial_data {
	std::vector<float> initial_radii;
	std::vector<float> initial_phis;
	std::vector<float> angular_momenta;

	shared_initial_data() = default;

	shared_initial_data(std::string_view file_path) {
		std::ifstream file(file_path.data());

		if (!file.is_open()) 
			throw std::runtime_error("File opened unsuccessfully");

		std::string line;
		std::string type;
		int i = 0;
		while (std::getline(file, line)) {
			std::stringstream ss(std::move(line));
			ss >> type;
			auto & vec = get_correct(i);
			while (!ss.eof()) {
				float x{};
				ss >> x;
				vec.push_back(x);
			}
			++i;
		}
	}

	std::vector<float>& get_correct(int i) {
		switch (i) {
		case 0:
			return initial_radii;
		case 1:
			return initial_phis;
		case 2:
			return angular_momenta;
		}

		return angular_momenta;
	}
};

template <spacetime metric>
struct initial_particle_data : public shared_initial_data {
	static const spacetime metric_type = metric;
	initial_particle_data(std::string_view file_path)
		: shared_initial_data(file_path) { }
};

template<>
struct initial_particle_data<spacetime::kerr> : public shared_initial_data {
	static const spacetime metric_type = spacetime::kerr;

	std::vector<float> initial_thetas;
	std::vector<float> carter_constants;
	std::vector<float> energies;
	std::vector<float> initial_p_r;
	std::vector<float> initial_p_theta;

	initial_particle_data(std::string_view file_path) {
		std::ifstream file(file_path.data());

		if (!file.is_open())
			throw std::runtime_error("File opened unsuccessfully");

		std::string line;
		std::string type;
		int i = 0;
		while (std::getline(file, line)) {
			std::stringstream ss(std::move(line));
			ss >> type;
			auto & vec = get_correct(i);
			while (!ss.eof()) {
				float x {};
				ss >> x;
				vec.push_back(x);
			}
			++i;
		}
	}

	std::vector<float> & get_correct(int i) {
		switch (i) {
		case 0:
			return initial_radii;
		case 1:
			return initial_phis;
		case 2:
			return angular_momenta;
		case 3:
			return initial_thetas;
		case 4:
			return carter_constants;
		case 5:
			return energies;
		case 6:
			return initial_p_r;
		case 7:
			return initial_p_theta;
		}
		return initial_p_theta;
	}
};

template <spacetime T>
float get_default(const initial_particle_data<T>& data) {
	if (spacetime::kerr == std::decay_t<decltype(data)>::metric_type)
		return 0.3f;
	else
		return 1.0f;
}

// MSCV does not define __FMA__ when AVX2 is enabled
#if defined(__AVX2__) && defined(_MSC_VER)
	#define __FMA__ 1
#endif

#ifdef __AVX512F__
	constexpr size_t subproblem_size = 16;

	#define ALIGN alignas(64)
	#define MFLOAT __m512
	#define MINT __m512i
	#define LOAD(x) _mm512_load_ps(x)
	#define STORE(x,y) _mm512_store_ps(x,y)
	#define STREAM_EPI32(x, y) _mm512_stream_si512(x, y)
	#define STREAM(x,y) _mm512_stream_ps(x, y)
	#define SET1(x) _mm512_set1_ps(x)
	#define SET1_EPI32(x) _mm512_set1_epi32(x)
	#define MUL(x, y) _mm512_mul_ps(x, y)
	#define FMADD(x, y, z) _mm512_fmadd_ps(x, y, z)
	#define FMSUB(x, y, z) _mm512_fmsub_ps(x, y, z)
	#define ADD(x, y) _mm512_add_ps(x, y)
	#define SUB(x, y) _mm512_sub_ps(x, y)
	#define SQRT(x) _mm512_sqrt_ps(x)
	#define COS(x) _mm512_cos_ps(x)
	#define SIN(x) _mm512_sin_ps(x)
	#define RCP(x) _mm512_rcp14_ps(x)
	#define SETEPI32(x) _mm512_set1_epi32(x)
	#define SETZERO _mm512_setzero_ps()
	#define SETZERO_EPI32 _mm512_setzero_si512()
	#define CMP(x, y, z) _mm512_cmp_ps_mask(x, y, z)
	#define MASKZ_MOV_EPI32(mask, x) _mm512_maskz_mov_epi32(mask, x)
	#define MASK_MUL(mask, x, y, z) _mm512_mask_mul_ps(x, mask, y, z)
	#define MASK_ADD(mas, x, y, z) _mm512_mask_add_ps(x, mask, y, z)
	#define ANDNOT(x, y) _mm512_andnot_ps(x, y)
	#define MAX(x, y) _mm512_max_ps(x, y)
	#define MIN(x, y) _mm512_min_ps(x, y)
	#define TESTZ(x, y) _ktestz_mask8_u8(x, y)

	template <typename T>
	void print_simd(T var) {
		using type = std::conditional_t<std::is_same_v<T, MFLOAT>, float, int>;
		type val[16];
		memcpy(val, &var, sizeof(val));
		printf("(%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f)\n",
			val[0], val[1], val[2], val[3], val[4], val[5],
			val[6], val[7], val[8], val[9], val[10], val[11],
			val[12], val[13], val[14], val[15]);
	}

#elif defined(__AVX2__)
	constexpr size_t subproblem_size = 8;

	#define ALIGN alignas(32)
	#define MFLOAT __m256
	#define MINT __m256i
	#define LOAD(x) _mm256_load_ps(x)
	#define STORE(x, y) _mm256_store_ps(x, y)
	#define STREAM_EPI32(x, y) _mm256_stream_si256(x, y)
	#define STREAM(x, y) _mm256_stream_ps(x, y)
	#define SET1(x) _mm256_set1_ps(x)
	#define SET1_EPI32(x) _mm256_set1_epi32(x)
	#define MUL(x, y) _mm256_mul_ps(x, y)
	#define ADD(x, y) _mm256_add_ps(x, y)
	#define SUB(x, y) _mm256_sub_ps(x, y)
	#ifdef __FMA__
		#define FMADD(x, y, z) _mm256_fmadd_ps(x, y, z)
		#define FMSUB(x, y, z) _mm256_fmsub_ps(x, y, z)
	#else
		#define FMADD(x, y, z) ADD(MUL(x,y), z)
		#define FMSUB(x, y, z) SUB(MUL(x, y), z)
	#endif
	#define SQRT(x) _mm256_sqrt_ps(x)
	#define COS(x) _mm256_cos_ps(x)
	#define SIN(x) _mm256_sin_ps(x)
	#define RCP(x) _mm256_rcp_ps(x)
	#define SETEPI32(x) _mm256_set1_epi32(x)
	#define SETZERO _mm256_setzero_ps()
	#define SETZERO_EPI32 _mm256_setzero_si256()
	#define CMP(x, y, z) _mm256_cmp_ps(x, y, z)
	#define MASKZ_MOV_EPI32(mask, x) _mm256_blendv_epi8(SETZERO_EPI32, x, mask)
	#define MASK_MUL(mask, x, y, z) _mm256_blendv_ps(x, MUL(y, z), mask)
	#define MASK_ADD(mask, x, y, z) _mm256_blendv_ps(x, ADD(y, z), mask)
	#define ANDNOT(x, y) _mm256_andnot_ps(x, y)
	#define MASK_MOVE(x, y, mask) _mm256_blendv_ps(x, y, mask)
	#define MAX(x, y) _mm256_max_ps(x, y)
	#define MIN(x, y) _mm256_min_ps(x, y)
	#define TESTZ(x, y) _mm256_testz_ps(x, y)

	template <typename T>
	void print_simd(T var) {
		using type = std::conditional_t<std::is_same_v<T, MFLOAT>, float, int>;
		type val[8];
		memcpy(val, &var, sizeof(val));
		printf("(%f, %f, %f, %f, %f, %f, %f, %f)\n",
			val[0], val[1], val[2], val[3], val[4], val[5],
			val[6], val[7]);
	}

#endif

template<size_t N>
MFLOAT _compute_L2_norm(const std::array<MFLOAT, N> & arr) {
	// first
	MFLOAT norm = MUL(arr[0], arr[0]);
	// sum the squares of the others
	for (int i = 1; i < N; ++i)
		norm = FMADD(arr[i], arr[i], norm);

	return SQRT(norm);
}

