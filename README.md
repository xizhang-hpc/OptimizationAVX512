# OptAVX512
Optimize CFD computing module by AVX512.
Unstructured mesh with indirect data access is considered.
## 1. compile
### 1.1 Data interpolation
icc *.cpp -DDATAINTERPOLATE -o dataInterp
### 1.2 Flux summation
icc *.cpp -DFLUXSUM -o fluxSum
### 1.3 Local pressure comparing
icc *.cpp -DMAXMIN -o minMax
## 2. Test
dataSet folder contains mesh data for Mesh 1 (unstructured mesh 1.05M cells and 2.32M faces).  
Taking Flux summation test for instance:  
tar -xzvf 100w_28.tgz  
cd 100w_28  
cp ../../fluxSum .  
./fluxSum  

## 3. Result
Taking results of flux summation for instance  
No.     Name            Parent     Elapsed Time    frequency  
1       HostFaceLoopLoadFluxCommonInside        HostFaceLoopLoadFlux    0.310000        500   
2       HostFaceLoopLoadFluxCommonOutside       HostFaceLoopLoadFlux    0.630000        500   
3       HostFaceLoopLoadAVXOutside      AVX512FaceLoopLoadFluxOutside   0.780000        500  
4       HostFaceLoopLoadAVXOutsideReOpt AVX512FaceLoopLoadFluxOutsideReOpt 0.380000      500   
5       HostCellLoopLoadCommonInside    HostCellLoopLoadFlux    0.750000        500   
6       HostCellLoopLoadCommonOutside   HostCellLoopLoadFlux    1.990000        500  
7       HostCellLoopLoadAVXOutside      AVX512CellLoopLoadFluxOutside   0.710000        500  

Elapsed Time shows the wall time on CPU. Frequency is the repeat times. Parent gives the function name that where loop is. Name is the explanation of the loop.

## 4. Loop relationship
Taking flux summation for instance  
Original function---AVX512 optimization  
HostFaceLoopLoadFluxCommonOutside---HostFaceLoopLoadAVXOutsideReOpt  
HostCellLoopLoadCommonOutside---HostCellLoopLoadAVXOutside



