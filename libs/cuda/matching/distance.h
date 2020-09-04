int32_t h_3d_cost_matrix[26][26] =
        {{0, 15, 31, 15, 15, 15, 9, 9, 21, 21, 9, 9, 21, 21, 7, 15, 23, 15, 7, 23, 23, 7, 7, 15, 23, 15},
{15, 0, 15, 31, 15, 15, 21, 9, 9, 21, 21, 9, 9, 21, 15, 7, 15, 23, 7, 7, 23, 23, 15, 23, 15, 7},
{31, 15, 0, 15, 15, 15, 21, 21, 9, 9, 21, 21, 9, 9, 23, 15, 7, 15, 23, 7, 7, 23, 23, 15, 7, 15},
{15, 31, 15, 0, 15, 15, 9, 21, 21, 9, 9, 21, 21, 9, 15, 23, 15, 7, 23, 23, 7, 7, 15, 7, 15, 23},
{15, 15, 15, 15, 0, 31, 21, 21, 21, 21, 9, 9, 9, 9, 23, 23, 23, 23, 15, 15, 15, 15, 7, 7, 7, 7},
{15, 15, 15, 15, 31, 0, 9, 9, 9, 9, 21, 21, 21, 21, 7, 7, 7, 7, 15, 15, 15, 15, 23, 23, 23, 23},
{9, 21, 21, 9, 21, 9, 0, 12, 19, 12, 12, 19, 31, 19, 6, 15, 15, 6, 15, 25, 15, 6, 15, 15, 25, 25},
{9, 9, 21, 21, 21, 9, 12, 0, 12, 19, 19, 12, 19, 31, 6, 6, 15, 15, 6, 15, 25, 15, 15, 25, 25, 15},
{21, 9, 9, 21, 21, 9, 19, 12, 0, 12, 31, 19, 12, 19, 15, 6, 6, 15, 15, 6, 15, 25, 25, 25, 15, 15},
{21, 21, 9, 9, 21, 9, 12, 19, 12, 0, 19, 31, 19, 12, 15, 15, 6, 6, 25, 15, 6, 15, 25, 15, 15, 25},
{9, 21, 21, 9, 9, 21, 12, 19, 31, 19, 0, 12, 19, 12, 15, 25, 25, 15, 15, 25, 15, 6, 6, 6, 15, 15},
{9, 9, 21, 21, 9, 21, 19, 12, 19, 31, 12, 0, 12, 19, 15, 15, 25, 25, 6, 15, 25, 15, 6, 15, 15, 6},
{21, 9, 9, 21, 9, 21, 31, 19, 12, 19, 19, 12, 0, 12, 25, 15, 15, 25, 15, 6, 15, 25, 15, 15, 6, 6},
{21, 21, 9, 9, 9, 21, 19, 31, 19, 12, 12, 19, 12, 0, 25, 25, 15, 15, 25, 15, 6, 15, 15, 6, 6, 15},
{7, 15, 23, 15, 23, 7, 6, 6, 15, 15, 15, 15, 25, 25, 0, 10, 15, 10, 10, 20, 20, 10, 15, 20, 31, 20},
{15, 7, 15, 23, 23, 7, 15, 6, 6, 15, 25, 15, 15, 25, 10, 0, 10, 15, 10, 10, 20, 20, 20, 31, 20, 15},
{23, 15, 7, 15, 23, 7, 15, 15, 6, 6, 25, 25, 15, 15, 15, 10, 0, 10, 20, 10, 10, 20, 31, 20, 15, 20},
{15, 23, 15, 7, 23, 7, 6, 15, 15, 6, 15, 25, 25, 15, 10, 15, 10, 0, 20, 20, 10, 10, 20, 15, 20, 31},
{7, 7, 23, 23, 15, 15, 15, 6, 15, 25, 15, 6, 15, 25, 10, 10, 20, 20, 0, 15, 31, 15, 10, 20, 20, 10},
{23, 7, 7, 23, 15, 15, 25, 15, 6, 15, 25, 15, 6, 15, 20, 10, 10, 20, 15, 0, 15, 31, 20, 20, 10, 10},
{23, 23, 7, 7, 15, 15, 15, 25, 15, 6, 15, 25, 15, 6, 20, 20, 10, 10, 31, 15, 0, 15, 20, 10, 10, 20},
{7, 23, 23, 7, 15, 15, 6, 15, 25, 15, 6, 15, 25, 15, 10, 20, 20, 10, 15, 31, 15, 0, 10, 10, 20, 20},
{7, 15, 23, 15, 7, 23, 15, 15, 25, 25, 6, 6, 15, 15, 15, 20, 31, 20, 10, 20, 20, 10, 0, 10, 15, 10},
{15, 23, 15, 7, 7, 23, 15, 25, 25, 15, 6, 15, 15, 6, 20, 31, 20, 15, 20, 20, 10, 10, 10, 0, 10, 15},
{23, 15, 7, 15, 7, 23, 25, 25, 15, 15, 15, 15, 6, 6, 31, 20, 15, 20, 20, 10, 10, 20, 15, 10, 0, 10},
{15, 7, 15, 23, 7, 23, 25, 15, 15, 25, 15, 6, 6, 15, 20, 15, 20, 31, 10, 10, 20, 20, 10, 15, 10, 0}};

int32_t h_2d_cost_matrix[8][8] = {{0, 15, 31, 15, 7, 23, 23, 7},
                                {15, 0, 15, 31, 7, 7, 23, 23},
                                {31, 15, 0, 15, 23, 7, 7, 23},
                                {15, 31, 15, 0, 23, 23, 7, 7},
                                {7, 7, 23, 23, 0, 15, 31, 15},
                                {23, 7, 7, 23, 15, 0, 15, 31},
                                {23, 23, 7, 7, 31, 15, 0, 15},
                                {7, 23, 23, 7, 15, 31, 15, 0}};
