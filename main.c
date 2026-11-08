/*
 * Программа логически работает следующим образом:
 * _mm_add_ps добавляет 4 упакованных числа с плавающей точкой одинарной точности в ОДНУ команду.
 * Мы вычисляем c[i] = a[i] + b[i] по массиву из n чисел с плавающей точкой, где n
 * намеренно НЕ кратно 4:
 * - основная часть обрабатывается по 4 элемента за раз с помощью SSE,
 * - оставшиеся 4 элемента ("хвост") обрабатываются скалярным циклом.
 * Выравнивание: если все три массива выровнены по 16 байтам, мы используем _mm_load_ps. /
 * _mm_store_ps, иначе _mm_loadu_ps / _mm_storeu_ps.
 * Результат проверяется по простому скалярному циклу и выводится максимальная
 * абсолютная ошибка (для сложения она должна быть точно равна 0, потому что
 * Добавление SSE с плавающей точкой соответствует стандарту IEEE-754, идентичному скалярному добавлению с плавающей точкой).
 *
 * Сборка: gcc -O2 -msse2 main.c -o main -lm (Makefile)
 */
#include <emmintrin.h>   /* SSE / SSE2 intrinsics */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define STUDENT_N 7
#define VARIANT   ((((STUDENT_N) - 1) % 24) + 1)
#define INSTR     "_mm_add_ps"

static int is_aligned16(const void *p) { return ((uintptr_t)p % 16) == 0; }

/* c[i] = a[i] + b[i] using _mm_add_ps; scalar tail for the remainder. */
static void add_simd(float *c, const float *a, const float *b, size_t n)
{
    size_t i = 0;
    size_t vec_n = n - (n % 4);   /* part divisible by 4 */

    if (is_aligned16(a) && is_aligned16(b) && is_aligned16(c)) {
        for (; i < vec_n; i += 4) {
            __m128 va = _mm_load_ps(a + i);
            __m128 vb = _mm_load_ps(b + i);
            _mm_store_ps(c + i, _mm_add_ps(va, vb));
        }
    } else {
        for (; i < vec_n; i += 4) {
            __m128 va = _mm_loadu_ps(a + i);
            __m128 vb = _mm_loadu_ps(b + i);
            _mm_storeu_ps(c + i, _mm_add_ps(va, vb));
        }
    }
    for (; i < n; i++)            /* scalar tail: n % 4 elements */
        c[i] = a[i] + b[i];
}

static void add_scalar(float *c, const float *a, const float *b, size_t n)
{
    for (size_t i = 0; i < n; i++)
        c[i] = a[i] + b[i];
}

/* Runs one test; returns max abs error. 'offset' shifts pointers to break alignment. */
static double run(const char *label, size_t n, size_t offset)
{
    size_t cap = ((n + offset + 3) / 4) * 4 * sizeof(float);
    float *A = aligned_alloc(16, cap), *B = aligned_alloc(16, cap);
    float *C = aligned_alloc(16, cap), *R = aligned_alloc(16, cap);
    if (!A || !B || !C || !R) { fprintf(stderr, "alloc failed\n"); exit(1); }

    float *a = A + offset, *b = B + offset, *c = C + offset, *r = R + offset;
    srand(12345);
    for (size_t i = 0; i < n; i++) {
        a[i] = (float)rand() / RAND_MAX * 200.0f - 100.0f;
        b[i] = (float)rand() / RAND_MAX * 200.0f - 100.0f;
    }

    add_simd(c, a, b, n);
    add_scalar(r, a, b, n);

    double max_err = 0.0, sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        double e = fabs((double)c[i] - (double)r[i]);
        if (e > max_err) max_err = e;
        sum += c[i];
    }

    printf("[%s] n=%zu (n %% 4 = %zu), aligned=%s\n", label, n, n % 4,
           (is_aligned16(a) && is_aligned16(b) && is_aligned16(c)) ? "yes" : "no");
    printf("  a[0]+b[0]   = %.4f + %.4f = %.4f\n", a[0], b[0], c[0]);
    printf("  a[n-1]+b[n-1] = %.4f + %.4f = %.4f (tail element)\n",
           a[n - 1], b[n - 1], c[n - 1]);
    printf("  sum(c) = %.4f\n", sum);
    printf("  max abs error vs scalar loop = %g\n", max_err);

    free(A); free(B); free(C); free(R);
    return max_err;
}

int main(void)
{
    printf("Student number: %d\n", STUDENT_N);
    printf("Variant: %d\n", VARIANT);
    printf("Instruction: %s\n\n", INSTR);

    double e1 = run("aligned load/store", 1003, 0);
    printf("\n");
    double e2 = run("unaligned loadu/storeu", 1003, 1);

    printf("\nResult: %s\n", (e1 == 0.0 && e2 == 0.0) ? "OK" : "MISMATCH");
    return (e1 == 0.0 && e2 == 0.0) ? 0 : 1;
}
