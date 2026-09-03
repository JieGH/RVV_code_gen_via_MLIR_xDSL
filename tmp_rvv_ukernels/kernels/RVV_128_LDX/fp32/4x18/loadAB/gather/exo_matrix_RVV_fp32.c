#include "exo_matrix_RVV_fp32.h"

ukrFunction**** allocateMatrix() {
    ukrFunction**** matrix = (ukrFunction****)malloc(5 * sizeof(ukrFunction***));
    for (int i = 0; i < 5; i++) {
        matrix[i] = (ukrFunction***)malloc(19 * sizeof(ukrFunction**));
        for (int j = 0; j < 19; j++) {
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
    *matrix[0][3][0] = 	(ukrFunction)NULL;
    *matrix[0][3][1] = 	(ukrFunction)NULL;
    *matrix[0][4][0] = 	(ukrFunction)NULL;
    *matrix[0][4][1] = 	(ukrFunction)NULL;
    *matrix[0][5][0] = 	(ukrFunction)NULL;
    *matrix[0][5][1] = 	(ukrFunction)NULL;
    *matrix[0][6][0] = 	(ukrFunction)NULL;
    *matrix[0][6][1] = 	(ukrFunction)NULL;
    *matrix[0][7][0] = 	(ukrFunction)NULL;
    *matrix[0][7][1] = 	(ukrFunction)NULL;
    *matrix[0][8][0] = 	(ukrFunction)NULL;
    *matrix[0][8][1] = 	(ukrFunction)NULL;
    *matrix[0][9][0] = 	(ukrFunction)NULL;
    *matrix[0][9][1] = 	(ukrFunction)NULL;
    *matrix[0][10][0] = 	(ukrFunction)NULL;
    *matrix[0][10][1] = 	(ukrFunction)NULL;
    *matrix[0][11][0] = 	(ukrFunction)NULL;
    *matrix[0][11][1] = 	(ukrFunction)NULL;
    *matrix[0][12][0] = 	(ukrFunction)NULL;
    *matrix[0][12][1] = 	(ukrFunction)NULL;
    *matrix[0][13][0] = 	(ukrFunction)NULL;
    *matrix[0][13][1] = 	(ukrFunction)NULL;
    *matrix[0][14][0] = 	(ukrFunction)NULL;
    *matrix[0][14][1] = 	(ukrFunction)NULL;
    *matrix[0][15][0] = 	(ukrFunction)NULL;
    *matrix[0][15][1] = 	(ukrFunction)NULL;
    *matrix[0][16][0] = 	(ukrFunction)NULL;
    *matrix[0][16][1] = 	(ukrFunction)NULL;
    *matrix[0][17][0] = 	(ukrFunction)NULL;
    *matrix[0][17][1] = 	(ukrFunction)NULL;
    *matrix[0][18][0] = 	(ukrFunction)NULL;
    *matrix[0][18][1] = 	(ukrFunction)NULL;
    *matrix[1][0][0] = 	(ukrFunction)NULL;
    *matrix[1][0][1] = 	(ukrFunction)NULL;
    *matrix[1][1][0] = 	(ukrFunction)gemm_RVV_1x1_b0_col_fp32;
    *matrix[1][1][1] = 	(ukrFunction)gemm_RVV_1x1_b1_col_fp32;
    *matrix[1][2][0] = 	(ukrFunction)gemm_RVV_1x2_b0_col_fp32;
    *matrix[1][2][1] = 	(ukrFunction)gemm_RVV_1x2_b1_col_fp32;
    *matrix[1][3][0] = 	(ukrFunction)gemm_RVV_1x3_b0_col_fp32;
    *matrix[1][3][1] = 	(ukrFunction)gemm_RVV_1x3_b1_col_fp32;
    *matrix[1][4][0] = 	(ukrFunction)gemm_RVV_1x4_b0_col_fp32;
    *matrix[1][4][1] = 	(ukrFunction)gemm_RVV_1x4_b1_col_fp32;
    *matrix[1][5][0] = 	(ukrFunction)gemm_RVV_1x5_b0_col_fp32;
    *matrix[1][5][1] = 	(ukrFunction)gemm_RVV_1x5_b1_col_fp32;
    *matrix[1][6][0] = 	(ukrFunction)gemm_RVV_1x6_b0_col_fp32;
    *matrix[1][6][1] = 	(ukrFunction)gemm_RVV_1x6_b1_col_fp32;
    *matrix[1][7][0] = 	(ukrFunction)gemm_RVV_1x7_b0_col_fp32;
    *matrix[1][7][1] = 	(ukrFunction)gemm_RVV_1x7_b1_col_fp32;
    *matrix[1][8][0] = 	(ukrFunction)gemm_RVV_1x8_b0_col_fp32;
    *matrix[1][8][1] = 	(ukrFunction)gemm_RVV_1x8_b1_col_fp32;
    *matrix[1][9][0] = 	(ukrFunction)gemm_RVV_1x9_b0_col_fp32;
    *matrix[1][9][1] = 	(ukrFunction)gemm_RVV_1x9_b1_col_fp32;
    *matrix[1][10][0] = 	(ukrFunction)gemm_RVV_1x10_b0_col_fp32;
    *matrix[1][10][1] = 	(ukrFunction)gemm_RVV_1x10_b1_col_fp32;
    *matrix[1][11][0] = 	(ukrFunction)gemm_RVV_1x11_b0_col_fp32;
    *matrix[1][11][1] = 	(ukrFunction)gemm_RVV_1x11_b1_col_fp32;
    *matrix[1][12][0] = 	(ukrFunction)gemm_RVV_1x12_b0_col_fp32;
    *matrix[1][12][1] = 	(ukrFunction)gemm_RVV_1x12_b1_col_fp32;
    *matrix[1][13][0] = 	(ukrFunction)gemm_RVV_1x13_b0_col_fp32;
    *matrix[1][13][1] = 	(ukrFunction)gemm_RVV_1x13_b1_col_fp32;
    *matrix[1][14][0] = 	(ukrFunction)gemm_RVV_1x14_b0_col_fp32;
    *matrix[1][14][1] = 	(ukrFunction)gemm_RVV_1x14_b1_col_fp32;
    *matrix[1][15][0] = 	(ukrFunction)gemm_RVV_1x15_b0_col_fp32;
    *matrix[1][15][1] = 	(ukrFunction)gemm_RVV_1x15_b1_col_fp32;
    *matrix[1][16][0] = 	(ukrFunction)gemm_RVV_1x16_b0_col_fp32;
    *matrix[1][16][1] = 	(ukrFunction)gemm_RVV_1x16_b1_col_fp32;
    *matrix[1][17][0] = 	(ukrFunction)gemm_RVV_1x17_b0_col_fp32;
    *matrix[1][17][1] = 	(ukrFunction)gemm_RVV_1x17_b1_col_fp32;
    *matrix[1][18][0] = 	(ukrFunction)gemm_RVV_1x18_b0_col_fp32;
    *matrix[1][18][1] = 	(ukrFunction)gemm_RVV_1x18_b1_col_fp32;
    *matrix[2][0][0] = 	(ukrFunction)NULL;
    *matrix[2][0][1] = 	(ukrFunction)NULL;
    *matrix[2][1][0] = 	(ukrFunction)gemm_RVV_2x1_b0_col_fp32;
    *matrix[2][1][1] = 	(ukrFunction)gemm_RVV_2x1_b1_col_fp32;
    *matrix[2][2][0] = 	(ukrFunction)gemm_RVV_2x2_b0_col_fp32;
    *matrix[2][2][1] = 	(ukrFunction)gemm_RVV_2x2_b1_col_fp32;
    *matrix[2][3][0] = 	(ukrFunction)gemm_RVV_2x3_b0_col_fp32;
    *matrix[2][3][1] = 	(ukrFunction)gemm_RVV_2x3_b1_col_fp32;
    *matrix[2][4][0] = 	(ukrFunction)gemm_RVV_2x4_b0_col_fp32;
    *matrix[2][4][1] = 	(ukrFunction)gemm_RVV_2x4_b1_col_fp32;
    *matrix[2][5][0] = 	(ukrFunction)gemm_RVV_2x5_b0_col_fp32;
    *matrix[2][5][1] = 	(ukrFunction)gemm_RVV_2x5_b1_col_fp32;
    *matrix[2][6][0] = 	(ukrFunction)gemm_RVV_2x6_b0_col_fp32;
    *matrix[2][6][1] = 	(ukrFunction)gemm_RVV_2x6_b1_col_fp32;
    *matrix[2][7][0] = 	(ukrFunction)gemm_RVV_2x7_b0_col_fp32;
    *matrix[2][7][1] = 	(ukrFunction)gemm_RVV_2x7_b1_col_fp32;
    *matrix[2][8][0] = 	(ukrFunction)gemm_RVV_2x8_b0_col_fp32;
    *matrix[2][8][1] = 	(ukrFunction)gemm_RVV_2x8_b1_col_fp32;
    *matrix[2][9][0] = 	(ukrFunction)gemm_RVV_2x9_b0_col_fp32;
    *matrix[2][9][1] = 	(ukrFunction)gemm_RVV_2x9_b1_col_fp32;
    *matrix[2][10][0] = 	(ukrFunction)gemm_RVV_2x10_b0_col_fp32;
    *matrix[2][10][1] = 	(ukrFunction)gemm_RVV_2x10_b1_col_fp32;
    *matrix[2][11][0] = 	(ukrFunction)gemm_RVV_2x11_b0_col_fp32;
    *matrix[2][11][1] = 	(ukrFunction)gemm_RVV_2x11_b1_col_fp32;
    *matrix[2][12][0] = 	(ukrFunction)gemm_RVV_2x12_b0_col_fp32;
    *matrix[2][12][1] = 	(ukrFunction)gemm_RVV_2x12_b1_col_fp32;
    *matrix[2][13][0] = 	(ukrFunction)gemm_RVV_2x13_b0_col_fp32;
    *matrix[2][13][1] = 	(ukrFunction)gemm_RVV_2x13_b1_col_fp32;
    *matrix[2][14][0] = 	(ukrFunction)gemm_RVV_2x14_b0_col_fp32;
    *matrix[2][14][1] = 	(ukrFunction)gemm_RVV_2x14_b1_col_fp32;
    *matrix[2][15][0] = 	(ukrFunction)gemm_RVV_2x15_b0_col_fp32;
    *matrix[2][15][1] = 	(ukrFunction)gemm_RVV_2x15_b1_col_fp32;
    *matrix[2][16][0] = 	(ukrFunction)gemm_RVV_2x16_b0_col_fp32;
    *matrix[2][16][1] = 	(ukrFunction)gemm_RVV_2x16_b1_col_fp32;
    *matrix[2][17][0] = 	(ukrFunction)gemm_RVV_2x17_b0_col_fp32;
    *matrix[2][17][1] = 	(ukrFunction)gemm_RVV_2x17_b1_col_fp32;
    *matrix[2][18][0] = 	(ukrFunction)gemm_RVV_2x18_b0_col_fp32;
    *matrix[2][18][1] = 	(ukrFunction)gemm_RVV_2x18_b1_col_fp32;
    *matrix[3][0][0] = 	(ukrFunction)NULL;
    *matrix[3][0][1] = 	(ukrFunction)NULL;
    *matrix[3][1][0] = 	(ukrFunction)gemm_RVV_3x1_b0_col_fp32;
    *matrix[3][1][1] = 	(ukrFunction)gemm_RVV_3x1_b1_col_fp32;
    *matrix[3][2][0] = 	(ukrFunction)gemm_RVV_3x2_b0_col_fp32;
    *matrix[3][2][1] = 	(ukrFunction)gemm_RVV_3x2_b1_col_fp32;
    *matrix[3][3][0] = 	(ukrFunction)gemm_RVV_3x3_b0_col_fp32;
    *matrix[3][3][1] = 	(ukrFunction)gemm_RVV_3x3_b1_col_fp32;
    *matrix[3][4][0] = 	(ukrFunction)gemm_RVV_3x4_b0_col_fp32;
    *matrix[3][4][1] = 	(ukrFunction)gemm_RVV_3x4_b1_col_fp32;
    *matrix[3][5][0] = 	(ukrFunction)gemm_RVV_3x5_b0_col_fp32;
    *matrix[3][5][1] = 	(ukrFunction)gemm_RVV_3x5_b1_col_fp32;
    *matrix[3][6][0] = 	(ukrFunction)gemm_RVV_3x6_b0_col_fp32;
    *matrix[3][6][1] = 	(ukrFunction)gemm_RVV_3x6_b1_col_fp32;
    *matrix[3][7][0] = 	(ukrFunction)gemm_RVV_3x7_b0_col_fp32;
    *matrix[3][7][1] = 	(ukrFunction)gemm_RVV_3x7_b1_col_fp32;
    *matrix[3][8][0] = 	(ukrFunction)gemm_RVV_3x8_b0_col_fp32;
    *matrix[3][8][1] = 	(ukrFunction)gemm_RVV_3x8_b1_col_fp32;
    *matrix[3][9][0] = 	(ukrFunction)gemm_RVV_3x9_b0_col_fp32;
    *matrix[3][9][1] = 	(ukrFunction)gemm_RVV_3x9_b1_col_fp32;
    *matrix[3][10][0] = 	(ukrFunction)gemm_RVV_3x10_b0_col_fp32;
    *matrix[3][10][1] = 	(ukrFunction)gemm_RVV_3x10_b1_col_fp32;
    *matrix[3][11][0] = 	(ukrFunction)gemm_RVV_3x11_b0_col_fp32;
    *matrix[3][11][1] = 	(ukrFunction)gemm_RVV_3x11_b1_col_fp32;
    *matrix[3][12][0] = 	(ukrFunction)gemm_RVV_3x12_b0_col_fp32;
    *matrix[3][12][1] = 	(ukrFunction)gemm_RVV_3x12_b1_col_fp32;
    *matrix[3][13][0] = 	(ukrFunction)gemm_RVV_3x13_b0_col_fp32;
    *matrix[3][13][1] = 	(ukrFunction)gemm_RVV_3x13_b1_col_fp32;
    *matrix[3][14][0] = 	(ukrFunction)gemm_RVV_3x14_b0_col_fp32;
    *matrix[3][14][1] = 	(ukrFunction)gemm_RVV_3x14_b1_col_fp32;
    *matrix[3][15][0] = 	(ukrFunction)gemm_RVV_3x15_b0_col_fp32;
    *matrix[3][15][1] = 	(ukrFunction)gemm_RVV_3x15_b1_col_fp32;
    *matrix[3][16][0] = 	(ukrFunction)gemm_RVV_3x16_b0_col_fp32;
    *matrix[3][16][1] = 	(ukrFunction)gemm_RVV_3x16_b1_col_fp32;
    *matrix[3][17][0] = 	(ukrFunction)gemm_RVV_3x17_b0_col_fp32;
    *matrix[3][17][1] = 	(ukrFunction)gemm_RVV_3x17_b1_col_fp32;
    *matrix[3][18][0] = 	(ukrFunction)gemm_RVV_3x18_b0_col_fp32;
    *matrix[3][18][1] = 	(ukrFunction)gemm_RVV_3x18_b1_col_fp32;
    *matrix[4][0][0] = 	(ukrFunction)NULL;
    *matrix[4][0][1] = 	(ukrFunction)NULL;
    *matrix[4][1][0] = 	(ukrFunction)gemm_RVV_4x1_b0_col_fp32;
    *matrix[4][1][1] = 	(ukrFunction)gemm_RVV_4x1_b1_col_fp32;
    *matrix[4][2][0] = 	(ukrFunction)gemm_RVV_4x2_b0_col_fp32;
    *matrix[4][2][1] = 	(ukrFunction)gemm_RVV_4x2_b1_col_fp32;
    *matrix[4][3][0] = 	(ukrFunction)gemm_RVV_4x3_b0_col_fp32;
    *matrix[4][3][1] = 	(ukrFunction)gemm_RVV_4x3_b1_col_fp32;
    *matrix[4][4][0] = 	(ukrFunction)gemm_RVV_4x4_b0_col_fp32;
    *matrix[4][4][1] = 	(ukrFunction)gemm_RVV_4x4_b1_col_fp32;
    *matrix[4][5][0] = 	(ukrFunction)gemm_RVV_4x5_b0_col_fp32;
    *matrix[4][5][1] = 	(ukrFunction)gemm_RVV_4x5_b1_col_fp32;
    *matrix[4][6][0] = 	(ukrFunction)gemm_RVV_4x6_b0_col_fp32;
    *matrix[4][6][1] = 	(ukrFunction)gemm_RVV_4x6_b1_col_fp32;
    *matrix[4][7][0] = 	(ukrFunction)gemm_RVV_4x7_b0_col_fp32;
    *matrix[4][7][1] = 	(ukrFunction)gemm_RVV_4x7_b1_col_fp32;
    *matrix[4][8][0] = 	(ukrFunction)gemm_RVV_4x8_b0_col_fp32;
    *matrix[4][8][1] = 	(ukrFunction)gemm_RVV_4x8_b1_col_fp32;
    *matrix[4][9][0] = 	(ukrFunction)gemm_RVV_4x9_b0_col_fp32;
    *matrix[4][9][1] = 	(ukrFunction)gemm_RVV_4x9_b1_col_fp32;
    *matrix[4][10][0] = 	(ukrFunction)gemm_RVV_4x10_b0_col_fp32;
    *matrix[4][10][1] = 	(ukrFunction)gemm_RVV_4x10_b1_col_fp32;
    *matrix[4][11][0] = 	(ukrFunction)gemm_RVV_4x11_b0_col_fp32;
    *matrix[4][11][1] = 	(ukrFunction)gemm_RVV_4x11_b1_col_fp32;
    *matrix[4][12][0] = 	(ukrFunction)gemm_RVV_4x12_b0_col_fp32;
    *matrix[4][12][1] = 	(ukrFunction)gemm_RVV_4x12_b1_col_fp32;
    *matrix[4][13][0] = 	(ukrFunction)gemm_RVV_4x13_b0_col_fp32;
    *matrix[4][13][1] = 	(ukrFunction)gemm_RVV_4x13_b1_col_fp32;
    *matrix[4][14][0] = 	(ukrFunction)gemm_RVV_4x14_b0_col_fp32;
    *matrix[4][14][1] = 	(ukrFunction)gemm_RVV_4x14_b1_col_fp32;
    *matrix[4][15][0] = 	(ukrFunction)gemm_RVV_4x15_b0_col_fp32;
    *matrix[4][15][1] = 	(ukrFunction)gemm_RVV_4x15_b1_col_fp32;
    *matrix[4][16][0] = 	(ukrFunction)gemm_RVV_4x16_b0_col_fp32;
    *matrix[4][16][1] = 	(ukrFunction)gemm_RVV_4x16_b1_col_fp32;
    *matrix[4][17][0] = 	(ukrFunction)gemm_RVV_4x17_b0_col_fp32;
    *matrix[4][17][1] = 	(ukrFunction)gemm_RVV_4x17_b1_col_fp32;
    *matrix[4][18][0] = 	(ukrFunction)gemm_RVV_4x18_b0_col_fp32;
    *matrix[4][18][1] = 	(ukrFunction)gemm_RVV_4x18_b1_col_fp32;
}


void freeMatrix(ukrFunction**** matrix) {
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 19; j++) {
            for (int b = 0; b < 2; b++) {
                free(matrix[i][j][b]);
            }
            free(matrix[i][j]);
        }
        free(matrix[i]);
    }
    free(matrix);
}


