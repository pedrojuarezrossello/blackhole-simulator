#pragma once
#include <array>
#include <type_traits>

namespace impl {
template <typename T, typename F, std::size_t... Is, typename ... Args>
constexpr std::array<std::decay_t<T>, sizeof...(Is)>
create_array(std::index_sequence<Is...>, F&& fn, Args&&... args) {
	return { { (static_cast<void>(Is), std::forward<F>(fn)(Is, std::forward<Args>(args)...))... } };
	}
}

template <typename T, std::size_t N, typename F, typename... Args>
constexpr std::array<std::decay_t<T>, N> create_array(F&& fn, Args&&... args) {
	return impl::create_array<T>(std::make_index_sequence<N>(), std::forward<F>(fn), std::forward<Args>(args)...);
}

template <size_t N>
struct initial_particle_data {
	std::array<float, N> initial_radii;
	std::array<float, N> initial_phis;
	std::array<float, N> initial_thetas;
	std::array<float, N> angular_momenta;
};

constexpr size_t N = 8;

// MSCV does not define __FMA__ when AVX2 is enabled
#if defined(__AVX2__) && defined(_MSC_VER)
	#define __FMA__ 1
#endif

#ifdef __AVX512F__
	template <size_t N>
	constexpr size_t subproblem_size = N <= 16 ? N : 16;

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
	// define MASK_ADD
	#define ANDNOT(x, y) _mm512_andnot_ps(x, y)
#elif defined(__AVX2__)
	template <size_t N>
	constexpr size_t subproblem_size = N <= 8 ? N : 8;

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
	#else
		#define FMADD(x, y, z) ADD(MUL(x,y), z)
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
#endif

template<typename T>
void print_ymm(T var) {
	using type = std::conditional_t<std::is_same_v<T,MFLOAT>, float, int>;
	type val[8];
	memcpy(val, &var, sizeof(val));
	printf("(%f, %f, %f, %f, %f, %f, %f, %f)\n",
			val[0], val[1], val[2], val[3], val[4], val[5],
			val[6], val[7]);
}
