#include "exo_matrix_RVV_fp32.h"

ukrFunction**** allocateMatrix() {
    ukrFunction**** matrix = (ukrFunction****)malloc(29 * sizeof(ukrFunction***));
    for (int i = 0; i < 29; i++) {
        matrix[i] = (ukrFunction***)malloc(3 * sizeof(ukrFunction**));
        for (int j = 0; j < 3; j++) {
            matrix[i][j] = (ukrFunction**)malloc(2 * sizeof(ukrFunction*));
            for (int b = 0; b < 2; b++) {
                matrix[i][j][b] = (ukrFunction*)malloc(1 * sizeof(ukrFunction));
            }
        }
    }
    return matrix;
}


void fillMatrix(ukrFunction**** matrix) {
    *matrix[0][0][0] = 	(ukrFunction)NULL;
    *matrix[0][0][1] = 	(ukrFunction)NULL;
    *matrix[0][1][0] = 	(ukrFunction)NULL;
    *matrix[0][1][1] = 	(ukrFunction)NULL;
    *matrix[0][2][0] = 	(ukrFunction)NULL;
    *matrix[0][2][1] = 	(ukrFunction)NULL;
    *matrix[1][0][0] = 	(ukrFunction)NULL;
    *matrix[1][0][1] = 	(ukrFunction)NULL;
    *matrix[1][1][0] = 	(ukrFunction)gemm_RVV_1x1_b0_col_fp32;
    *matrix[1][1][1] = 	(ukrFunction)gemm_RVV_1x1_b1_col_fp32;
    *matrix[1][2][0] = 	(ukrFunction)gemm_RVV_1x2_b0_col_fp32;
    *matrix[1][2][1] = 	(ukrFunction)gemm_RVV_1x2_b1_col_fp32;
    *matrix[2][0][0] = 	(ukrFunction)NULL;
    *matrix[2][0][1] = 	(ukrFunction)NULL;
    *matrix[2][1][0] = 	(ukrFunction)gemm_RVV_2x1_b0_col_fp32;
    *matrix[2][1][1] = 	(ukrFunction)gemm_RVV_2x1_b1_col_fp32;
    *matrix[2][2][0] = 	(ukrFunction)gemm_RVV_2x2_b0_col_fp32;
    *matrix[2][2][1] = 	(ukrFunction)gemm_RVV_2x2_b1_col_fp32;
    *matrix[3][0][0] = 	(ukrFunction)NULL;
    *matrix[3][0][1] = 	(ukrFunction)NULL;
    *matrix[3][1][0] = 	(ukrFunction)gemm_RVV_3x1_b0_col_fp32;
    *matrix[3][1][1] = 	(ukrFunction)gemm_RVV_3x1_b1_col_fp32;
    *matrix[3][2][0] = 	(ukrFunction)gemm_RVV_3x2_b0_col_fp32;
    *matrix[3][2][1] = 	(ukrFunction)gemm_RVV_3x2_b1_col_fp32;
    *matrix[4][0][0] = 	(ukrFunction)NULL;
    *matrix[4][0][1] = 	(ukrFunction)NULL;
    *matrix[4][1][0] = 	(ukrFunction)gemm_RVV_4x1_b0_col_fp32;
    *matrix[4][1][1] = 	(ukrFunction)gemm_RVV_4x1_b1_col_fp32;
    *matrix[4][2][0] = 	(ukrFunction)gemm_RVV_4x2_b0_col_fp32;
    *matrix[4][2][1] = 	(ukrFunction)gemm_RVV_4x2_b1_col_fp32;
    *matrix[5][0][0] = 	(ukrFunction)NULL;
    *matrix[5][0][1] = 	(ukrFunction)NULL;
    *matrix[5][1][0] = 	(ukrFunction)gemm_RVV_5x1_b0_col_fp32;
    *matrix[5][1][1] = 	(ukrFunction)gemm_RVV_5x1_b1_col_fp32;
    *matrix[5][2][0] = 	(ukrFunction)gemm_RVV_5x2_b0_col_fp32;
    *matrix[5][2][1] = 	(ukrFunction)gemm_RVV_5x2_b1_col_fp32;
    *matrix[6][0][0] = 	(ukrFunction)NULL;
    *matrix[6][0][1] = 	(ukrFunction)NULL;
    *matrix[6][1][0] = 	(ukrFunction)gemm_RVV_6x1_b0_col_fp32;
    *matrix[6][1][1] = 	(ukrFunction)gemm_RVV_6x1_b1_col_fp32;
    *matrix[6][2][0] = 	(ukrFunction)gemm_RVV_6x2_b0_col_fp32;
    *matrix[6][2][1] = 	(ukrFunction)gemm_RVV_6x2_b1_col_fp32;
    *matrix[7][0][0] = 	(ukrFunction)NULL;
    *matrix[7][0][1] = 	(ukrFunction)NULL;
    *matrix[7][1][0] = 	(ukrFunction)gemm_RVV_7x1_b0_col_fp32;
    *matrix[7][1][1] = 	(ukrFunction)gemm_RVV_7x1_b1_col_fp32;
    *matrix[7][2][0] = 	(ukrFunction)gemm_RVV_7x2_b0_col_fp32;
    *matrix[7][2][1] = 	(ukrFunction)gemm_RVV_7x2_b1_col_fp32;
    *matrix[8][0][0] = 	(ukrFunction)NULL;
    *matrix[8][0][1] = 	(ukrFunction)NULL;
    *matrix[8][1][0] = 	(ukrFunction)gemm_RVV_8x1_b0_col_fp32;
    *matrix[8][1][1] = 	(ukrFunction)gemm_RVV_8x1_b1_col_fp32;
    *matrix[8][2][0] = 	(ukrFunction)gemm_RVV_8x2_b0_col_fp32;
    *matrix[8][2][1] = 	(ukrFunction)gemm_RVV_8x2_b1_col_fp32;
    *matrix[9][0][0] = 	(ukrFunction)NULL;
    *matrix[9][0][1] = 	(ukrFunction)NULL;
    *matrix[9][1][0] = 	(ukrFunction)gemm_RVV_9x1_b0_col_fp32;
    *matrix[9][1][1] = 	(ukrFunction)gemm_RVV_9x1_b1_col_fp32;
    *matrix[9][2][0] = 	(ukrFunction)gemm_RVV_9x2_b0_col_fp32;
    *matrix[9][2][1] = 	(ukrFunction)gemm_RVV_9x2_b1_col_fp32;
    *matrix[10][0][0] = 	(ukrFunction)NULL;
    *matrix[10][0][1] = 	(ukrFunction)NULL;
    *matrix[10][1][0] = 	(ukrFunction)gemm_RVV_10x1_b0_col_fp32;
    *matrix[10][1][1] = 	(ukrFunction)gemm_RVV_10x1_b1_col_fp32;
    *matrix[10][2][0] = 	(ukrFunction)gemm_RVV_10x2_b0_col_fp32;
    *matrix[10][2][1] = 	(ukrFunction)gemm_RVV_10x2_b1_col_fp32;
    *matrix[11][0][0] = 	(ukrFunction)NULL;
    *matrix[11][0][1] = 	(ukrFunction)NULL;
    *matrix[11][1][0] = 	(ukrFunction)gemm_RVV_11x1_b0_col_fp32;
    *matrix[11][1][1] = 	(ukrFunction)gemm_RVV_11x1_b1_col_fp32;
    *matrix[11][2][0] = 	(ukrFunction)gemm_RVV_11x2_b0_col_fp32;
    *matrix[11][2][1] = 	(ukrFunction)gemm_RVV_11x2_b1_col_fp32;
    *matrix[12][0][0] = 	(ukrFunction)NULL;
    *matrix[12][0][1] = 	(ukrFunction)NULL;
    *matrix[12][1][0] = 	(ukrFunction)gemm_RVV_12x1_b0_col_fp32;
    *matrix[12][1][1] = 	(ukrFunction)gemm_RVV_12x1_b1_col_fp32;
    *matrix[12][2][0] = 	(ukrFunction)gemm_RVV_12x2_b0_col_fp32;
    *matrix[12][2][1] = 	(ukrFunction)gemm_RVV_12x2_b1_col_fp32;
    *matrix[13][0][0] = 	(ukrFunction)NULL;
    *matrix[13][0][1] = 	(ukrFunction)NULL;
    *matrix[13][1][0] = 	(ukrFunction)gemm_RVV_13x1_b0_col_fp32;
    *matrix[13][1][1] = 	(ukrFunction)gemm_RVV_13x1_b1_col_fp32;
    *matrix[13][2][0] = 	(ukrFunction)gemm_RVV_13x2_b0_col_fp32;
    *matrix[13][2][1] = 	(ukrFunction)gemm_RVV_13x2_b1_col_fp32;
    *matrix[14][0][0] = 	(ukrFunction)NULL;
    *matrix[14][0][1] = 	(ukrFunction)NULL;
    *matrix[14][1][0] = 	(ukrFunction)gemm_RVV_14x1_b0_col_fp32;
    *matrix[14][1][1] = 	(ukrFunction)gemm_RVV_14x1_b1_col_fp32;
    *matrix[14][2][0] = 	(ukrFunction)gemm_RVV_14x2_b0_col_fp32;
    *matrix[14][2][1] = 	(ukrFunction)gemm_RVV_14x2_b1_col_fp32;
    *matrix[15][0][0] = 	(ukrFunction)NULL;
    *matrix[15][0][1] = 	(ukrFunction)NULL;
    *matrix[15][1][0] = 	(ukrFunction)gemm_RVV_15x1_b0_col_fp32;
    *matrix[15][1][1] = 	(ukrFunction)gemm_RVV_15x1_b1_col_fp32;
    *matrix[15][2][0] = 	(ukrFunction)gemm_RVV_15x2_b0_col_fp32;
    *matrix[15][2][1] = 	(ukrFunction)gemm_RVV_15x2_b1_col_fp32;
    *matrix[16][0][0] = 	(ukrFunction)NULL;
    *matrix[16][0][1] = 	(ukrFunction)NULL;
    *matrix[16][1][0] = 	(ukrFunction)gemm_RVV_16x1_b0_col_fp32;
    *matrix[16][1][1] = 	(ukrFunction)gemm_RVV_16x1_b1_col_fp32;
    *matrix[16][2][0] = 	(ukrFunction)gemm_RVV_16x2_b0_col_fp32;
    *matrix[16][2][1] = 	(ukrFunction)gemm_RVV_16x2_b1_col_fp32;
    *matrix[17][0][0] = 	(ukrFunction)NULL;
    *matrix[17][0][1] = 	(ukrFunction)NULL;
    *matrix[17][1][0] = 	(ukrFunction)gemm_RVV_17x1_b0_col_fp32;
    *matrix[17][1][1] = 	(ukrFunction)gemm_RVV_17x1_b1_col_fp32;
    *matrix[17][2][0] = 	(ukrFunction)gemm_RVV_17x2_b0_col_fp32;
    *matrix[17][2][1] = 	(ukrFunction)gemm_RVV_17x2_b1_col_fp32;
    *matrix[18][0][0] = 	(ukrFunction)NULL;
    *matrix[18][0][1] = 	(ukrFunction)NULL;
    *matrix[18][1][0] = 	(ukrFunction)gemm_RVV_18x1_b0_col_fp32;
    *matrix[18][1][1] = 	(ukrFunction)gemm_RVV_18x1_b1_col_fp32;
    *matrix[18][2][0] = 	(ukrFunction)gemm_RVV_18x2_b0_col_fp32;
    *matrix[18][2][1] = 	(ukrFunction)gemm_RVV_18x2_b1_col_fp32;
    *matrix[19][0][0] = 	(ukrFunction)NULL;
    *matrix[19][0][1] = 	(ukrFunction)NULL;
    *matrix[19][1][0] = 	(ukrFunction)gemm_RVV_19x1_b0_col_fp32;
    *matrix[19][1][1] = 	(ukrFunction)gemm_RVV_19x1_b1_col_fp32;
    *matrix[19][2][0] = 	(ukrFunction)gemm_RVV_19x2_b0_col_fp32;
    *matrix[19][2][1] = 	(ukrFunction)gemm_RVV_19x2_b1_col_fp32;
    *matrix[20][0][0] = 	(ukrFunction)NULL;
    *matrix[20][0][1] = 	(ukrFunction)NULL;
    *matrix[20][1][0] = 	(ukrFunction)gemm_RVV_20x1_b0_col_fp32;
    *matrix[20][1][1] = 	(ukrFunction)gemm_RVV_20x1_b1_col_fp32;
    *matrix[20][2][0] = 	(ukrFunction)gemm_RVV_20x2_b0_col_fp32;
    *matrix[20][2][1] = 	(ukrFunction)gemm_RVV_20x2_b1_col_fp32;
    *matrix[21][0][0] = 	(ukrFunction)NULL;
    *matrix[21][0][1] = 	(ukrFunction)NULL;
    *matrix[21][1][0] = 	(ukrFunction)gemm_RVV_21x1_b0_col_fp32;
    *matrix[21][1][1] = 	(ukrFunction)gemm_RVV_21x1_b1_col_fp32;
    *matrix[21][2][0] = 	(ukrFunction)gemm_RVV_21x2_b0_col_fp32;
    *matrix[21][2][1] = 	(ukrFunction)gemm_RVV_21x2_b1_col_fp32;
    *matrix[22][0][0] = 	(ukrFunction)NULL;
    *matrix[22][0][1] = 	(ukrFunction)NULL;
    *matrix[22][1][0] = 	(ukrFunction)gemm_RVV_22x1_b0_col_fp32;
    *matrix[22][1][1] = 	(ukrFunction)gemm_RVV_22x1_b1_col_fp32;
    *matrix[22][2][0] = 	(ukrFunction)gemm_RVV_22x2_b0_col_fp32;
    *matrix[22][2][1] = 	(ukrFunction)gemm_RVV_22x2_b1_col_fp32;
    *matrix[23][0][0] = 	(ukrFunction)NULL;
    *matrix[23][0][1] = 	(ukrFunction)NULL;
    *matrix[23][1][0] = 	(ukrFunction)gemm_RVV_23x1_b0_col_fp32;
    *matrix[23][1][1] = 	(ukrFunction)gemm_RVV_23x1_b1_col_fp32;
    *matrix[23][2][0] = 	(ukrFunction)gemm_RVV_23x2_b0_col_fp32;
    *matrix[23][2][1] = 	(ukrFunction)gemm_RVV_23x2_b1_col_fp32;
    *matrix[24][0][0] = 	(ukrFunction)NULL;
    *matrix[24][0][1] = 	(ukrFunction)NULL;
    *matrix[24][1][0] = 	(ukrFunction)gemm_RVV_24x1_b0_col_fp32;
    *matrix[24][1][1] = 	(ukrFunction)gemm_RVV_24x1_b1_col_fp32;
    *matrix[24][2][0] = 	(ukrFunction)gemm_RVV_24x2_b0_col_fp32;
    *matrix[24][2][1] = 	(ukrFunction)gemm_RVV_24x2_b1_col_fp32;
    *matrix[25][0][0] = 	(ukrFunction)NULL;
    *matrix[25][0][1] = 	(ukrFunction)NULL;
    *matrix[25][1][0] = 	(ukrFunction)gemm_RVV_25x1_b0_col_fp32;
    *matrix[25][1][1] = 	(ukrFunction)gemm_RVV_25x1_b1_col_fp32;
    *matrix[25][2][0] = 	(ukrFunction)gemm_RVV_25x2_b0_col_fp32;
    *matrix[25][2][1] = 	(ukrFunction)gemm_RVV_25x2_b1_col_fp32;
    *matrix[26][0][0] = 	(ukrFunction)NULL;
    *matrix[26][0][1] = 	(ukrFunction)NULL;
    *matrix[26][1][0] = 	(ukrFunction)gemm_RVV_26x1_b0_col_fp32;
    *matrix[26][1][1] = 	(ukrFunction)gemm_RVV_26x1_b1_col_fp32;
    *matrix[26][2][0] = 	(ukrFunction)gemm_RVV_26x2_b0_col_fp32;
    *matrix[26][2][1] = 	(ukrFunction)gemm_RVV_26x2_b1_col_fp32;
    *matrix[27][0][0] = 	(ukrFunction)NULL;
    *matrix[27][0][1] = 	(ukrFunction)NULL;
    *matrix[27][1][0] = 	(ukrFunction)gemm_RVV_27x1_b0_col_fp32;
    *matrix[27][1][1] = 	(ukrFunction)gemm_RVV_27x1_b1_col_fp32;
    *matrix[27][2][0] = 	(ukrFunction)gemm_RVV_27x2_b0_col_fp32;
    *matrix[27][2][1] = 	(ukrFunction)gemm_RVV_27x2_b1_col_fp32;
    *matrix[28][0][0] = 	(ukrFunction)NULL;
    *matrix[28][0][1] = 	(ukrFunction)NULL;
    *matrix[28][1][0] = 	(ukrFunction)gemm_RVV_28x1_b0_col_fp32;
    *matrix[28][1][1] = 	(ukrFunction)gemm_RVV_28x1_b1_col_fp32;
    *matrix[28][2][0] = 	(ukrFunction)gemm_RVV_28x2_b0_col_fp32;
    *matrix[28][2][1] = 	(ukrFunction)gemm_RVV_28x2_b1_col_fp32;
}


void freeMatrix(ukrFunction**** matrix) {
    for (int i = 0; i < 29; i++) {
        for (int j = 0; j < 3; j++) {
            for (int b = 0; b < 2; b++) {
                free(matrix[i][j][b]);
            }
            free(matrix[i][j]);
        }
        free(matrix[i]);
    }
    free(matrix);
}


