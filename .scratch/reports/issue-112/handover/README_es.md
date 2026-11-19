# Microkernels xDSL para GEMMLib_yolo (issue #112)

Estos son nuestros microkernels RVV generados con xDSL. Cada uno va dentro de una función envoltorio que recibe los mismos argumentos que tus `uk_intrinsic_fp32` / `uk_intrinsic_fp16` (GEMMLib_yolo `b292c92`, `src/GEMM/gemm_blis.h`), así que tu GEMM puede llamarlos en lugar de tus propios kernels.

## Ficheros

| fichero | contenido |
|---|---|
| `xdsl_ukernels_fp32.{c,h}` | A, B y C en FP32. `xdsl_ukernel_<MR>x<NR>_fp32` |
| `xdsl_ukernels_f16f16f32.{c,h}` | A y B en FP16; C y todas las sumas en FP32. `xdsl_ukernel_<MR>x<NR>_f16f16f32` |
| `ref_ukernels.{c,h}` | referencia sencilla: bucles simples, sin código vectorial, mismos argumentos |
| `test_xdsl_ukernels.c` | comprueba cada kernel frente a la referencia |
| `Makefile` | compila y ejecuta la comprobación |

Tiles: MR = 8, 16, 24, 32 y NR = 1 … 16, es decir, 64 kernels por precisión.

## Argumentos (los mismos que los tuyos)

```c
void xdsl_ukernel_24x8_fp32     (int mr, int nr, int kc, const float    *Ar, const float    *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
void xdsl_ukernel_24x8_f16f16f32(int mr, int nr, int kc, const _Float16 *Ar, const _Float16 *Br, float *Cr, float beta, int ldA, int ldB, int Clda);
```

- `_Float16` es el mismo tipo que tu `fp16_t`.
- `Ar`: A empaquetada, MR valores por paso de k. `Br`: B empaquetada, NR valores por paso de k. `Cr`: por columnas, `Cr[j*Clda + i]`.
- `mr`, `nr`, `ldA` y `ldB` no se usan. Siempre se calcula el tile completo MR × NR, como en tus kernels; los bordes pasan por tu camino `Ctmp`.
- `beta == 0`: C = A·B. Cualquier otro `beta`: C += A·B (el valor de beta no se usa). No hay alpha.
- Solo se exportan los nombres `xdsl_ukernel_*`. Los kernels internos son `static`, así que nada choca con tus `ukernel_*`.

Para usar uno, ponlo donde tu GEMM elige el kernel (`src/GEMM/gemm_blis.c`, líneas 858-863 y 972-977 en `b292c92`), con `MR` y `NR` a juego. Por ejemplo: `MR=24; NR=8; uk = xdsl_ukernel_24x8_fp32;`.

**FP16: los resultados no coincidirán con tus kernels por defecto.** Con `NQ_FP16` tu GEMM usa `ukernel_32x8_fp16` y `ukernel_32x4_fp16`, que suman en FP16 (`vfmacc_vf_f16m2`) y solo pasan a FP32 al guardar C. Nuestros kernels `f16f16f32` suman en FP32 (`vfwmacc_vf_f32m1`), como tus kernels FP16 `16x8` y `16x4`. Nuestros resultados estarán más cerca de una referencia FP32. Una diferencia con tus resultados FP16 de 32 filas no es un error.

## Compilar y comprobar

En una placa K1 (RVV 1.0, VLEN 256, Zvfh):

```
make run            # gcc-14 -O3 -march=rv64gcv_zfh_zvfh
```

Usa gcc-14. gcc 13 no declara las intrínsecas FP16 de ensanchamiento, así que no compila el fichero FP16. El fichero FP32 compila con los dos.

La comprobación llama a cada kernel y a la referencia a través de tus propios tipos `uk_intrinsic_fp32` / `uk_intrinsic_fp16`, así que una firma incorrecta no compila. Cada kernel se ejecuta con:

- `kc` = 1, 7, 256;
- `beta` = 0, 1, 0.5;
- `Clda` = MR + 3 (las 3 filas de más no deben cambiar).

Hace dos pasadas de datos:

- **exacta:** enteros −5…5. Los resultados son exactos, así que deben coincidir bit a bit con la referencia.
- **aleatoria:** valores en [0, 1). El error relativo ‖C − ref‖ / ‖ref‖ debe ser ≤ max(1e-5, kc·1e-6).

Salida esperada:

```
fp32      : 64 kernels, 1152 calls, 0 kernels failed -> PASS
f16f16f32 : 64 kernels, 1152 calls, 0 kernels failed -> PASS
```

## Límites conocidos

- **Solo corrección:** en esta entrega los kernels no están ajustados para velocidad.
- **Presión de registros:** nuestros kernels guardan C en registros FP32 m1, así que un tile necesita MR/8 × NR registros para C más MR/8 para A. Los tiles de más de 32 registros hacen spill: MR = 32 con NR ≥ 8, MR = 24 con NR ≥ 10, y MR = 16 con NR = 16. Siguen siendo correctos, pero van lentos.
