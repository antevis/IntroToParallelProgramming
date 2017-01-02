include "reference_calc.cpp"
include "utils.h"

global
void gaussian_blur(const unsigned char* const inputChannel,
unsigned char* const outputChannel,
int numRows, int numCols,
const float* const filter, const int filterWidth)
{
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;

if ( col >= numCols || row >= numRows )
{
return;
}

float weightedPixel = 0.0f;
for (int filter_col = -filterWidth/2; filter_col <= filterWidth/2; ++filter_col) 
{
for (int filter_row = -filterWidth/2; filter_row <= filterWidth/2; ++filter_row) 
{
int image_row = min(max(row + filter_row, 0), numRows - 1);
int image_col = min(max(col + filter_col, 0), numCols - 1);
float image_value = inputChannel[image_row * numCols + image_col];
float filter_value = filter[(filter_row + filterWidth/2) * filterWidth + (filter_col + filterWidth/2)];
weightedPixel += image_value * filter_value;
}
}
outputChannel[row * numCols + col] = weightedPixel;
}

global
void separateChannels(const uchar4* const inputImageRGBA,
int numRows,
int numCols,
unsigned char* const redChannel,
unsigned char* const greenChannel,
unsigned char* const blueChannel)
{

const int2 thread_2D_pos = make_int2(threadIdx.x + blockIdx.x* blockDim.x, threadIdx.y + blockIdx.y * blockDim.y);

const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

unsigned char red;
unsigned char blue;
unsigned char green;
if(thread_2D_pos.x > numCols || thread_2D_pos.y > numRows)
return;

red = inputImageRGBA[thread_1D_pos].x;
green = inputImageRGBA[thread_1D_pos].y;
blue = inputImageRGBA[thread_1D_pos].z;

redChannel[thread_1D_pos] = red;
greenChannel[thread_1D_pos] = green;
blueChannel[thread_1D_pos] = blue;
}

global
void recombineChannels(const unsigned char* const redChannel,
const unsigned char* const greenChannel,
const unsigned char* const blueChannel,
uchar4* const outputImageRGBA,
int numRows,
int numCols)
{
const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
blockIdx.y * blockDim.y + threadIdx.y);

const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

if (thread_2D_pos.x > numCols || thread_2D_pos.y > numRows)
return;

unsigned char red = redChannel[thread_1D_pos];
unsigned char green = greenChannel[thread_1D_pos];
unsigned char blue = blueChannel[thread_1D_pos];

uchar4 outputPixel = make_uchar4(red, green, blue, 255);

outputImageRGBA[thread_1D_pos] = outputPixel;
}

unsigned char *d_red, *d_green, *d_blue;
float *d_filter;

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
const float* const h_filter, const size_t filterWidth)
{

checkCudaErrors(cudaMalloc(&d_red, sizeof(unsigned char) * numRowsImage * numColsImage));
checkCudaErrors(cudaMalloc(&d_green, sizeof(unsigned char) * numRowsImage * numColsImage));
checkCudaErrors(cudaMalloc(&d_blue, sizeof(unsigned char) * numRowsImage * numColsImage));

checkCudaErrors(cudaMalloc(&d_filter, sizeof(float) * filterWidth * filterWidth));

checkCudaErrors(cudaMemcpy(d_filter, h_filter, sizeof(float) * filterWidth * filterWidth, cudaMemcpyHostToDevice));

}

void your_gaussian_blur(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols,
unsigned char *d_redBlurred, 
unsigned char *d_greenBlurred, 
unsigned char *d_blueBlurred,
const int filterWidth)
{
const dim3 blockSize(16,16,1);
int a=numCols/blockSize.x, b=numRows/blockSize.y;

const dim3 gridSize(a+1,b+1,1);

separateChannels<<>>(d_inputImageRGBA,
numRows,
numCols,
d_red,
d_green,
d_blue);
cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

gaussian_blur<<>>(d_red, d_redBlurred, numRows, numCols, d_filter, filterWidth);
gaussian_blur<<>>(d_green, d_greenBlurred, numRows, numCols, d_filter, filterWidth);
gaussian_blur<<>>(d_blue, d_blueBlurred, numRows, numCols, d_filter, filterWidth);

cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

recombineChannels<<>>(d_redBlurred,
d_greenBlurred,
d_blueBlurred,
d_outputImageRGBA,
numRows,
numCols);
cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}

void cleanup() {
checkCudaErrors(cudaFree(d_red));
checkCudaErrors(cudaFree(d_green));
checkCudaErrors(cudaFree(d_blue));
checkCudaErrors(cudaFree(d_filter));
}
