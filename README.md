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
dataSet folder contains mesh data for Mesh 1 (1.05M cells and 2.32M faces).  
tar -xzvf 100w_28.tgz  
cd 100w_28  
cp ../../dataInterp .  
./dataInterp  
